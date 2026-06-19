// Typed client for the FastAPI agent backend. The wire contract mirrors
// apps/agent/api/schema (REST) and the SSE events emitted by /ask/stream.

export interface TraceStep {
  step: number;
  agent: string; // "planner" | "router" | "retriever" | "verifier" | "synthesizer"
  action: string; // "plan" | "route" | "read" | "verify" | "answer"
  arg: string;
  thought: string;
}

export interface Health {
  status: string; // "ok" | "degraded"
  wiki_pages: number;
  index_loaded: boolean;
  llm: { url: string; model: string; reachable: boolean };
}

export async function getHealth(): Promise<Health> {
  const r = await fetch("/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export interface DatasetsInfo {
  default: string; // the default dataset id
  datasets: Record<string, boolean>; // id -> built?
}

// Registered wiki datasets (versions) and whether each is built. Feeds the version picker.
export async function getDatasets(): Promise<DatasetsInfo> {
  const r = await fetch("/datasets");
  if (!r.ok) throw new Error(`datasets ${r.status}`);
  return r.json();
}

export interface StreamHandlers {
  onStep?: (s: TraceStep) => void;
  onToken?: (text: string) => void;
  onAnswer?: (answer: string, steps: number) => void;
  onError?: (detail: string) => void;
  onDone?: () => void;
}

/**
 * Open an SSE stream to /ask/stream and dispatch events. The backend emits:
 *   event: step   -> TraceStep
 *   event: token  -> { text }            // one answer fragment, streamed in order
 *   event: answer -> { answer, steps }   // the joined final answer (last)
 *   event: error  -> { detail }
 *   event: done   -> {}
 * Returns a cancel function that closes the EventSource.
 */
export function askStream(
  question: string,
  maxSteps: number,
  h: StreamHandlers,
  dataset?: string,
): () => void {
  let url = `/ask/stream?question=${encodeURIComponent(question)}&max_steps=${maxSteps}`;
  if (dataset) url += `&dataset=${encodeURIComponent(dataset)}`;
  const es = new EventSource(url);

  es.addEventListener("step", (e) => {
    h.onStep?.(JSON.parse((e as MessageEvent).data) as TraceStep);
  });
  es.addEventListener("token", (e) => {
    const d = JSON.parse((e as MessageEvent).data) as { text: string };
    h.onToken?.(d.text);
  });
  es.addEventListener("answer", (e) => {
    const d = JSON.parse((e as MessageEvent).data) as { answer: string; steps: number };
    h.onAnswer?.(d.answer, d.steps);
  });
  es.addEventListener("error", (e) => {
    // EventSource fires a bare "error" on network drop too (no data payload)
    let detail = "스트림 연결이 끊겼습니다.";
    const data = (e as MessageEvent).data;
    if (data) {
      try {
        detail = (JSON.parse(data) as { detail?: string }).detail ?? detail;
      } catch {
        /* keep default */
      }
    }
    es.close();
    h.onError?.(detail);
  });
  es.addEventListener("done", () => {
    es.close();
    h.onDone?.();
  });

  return () => es.close();
}
