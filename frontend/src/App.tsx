import { useCallback, useEffect, useRef, useState } from "react";
import { askStream, type TraceStep } from "./api";
import StatusBadge from "./components/StatusBadge";
import DatasetPicker from "./components/DatasetPicker";
import Composer from "./components/Composer";
import ChatMessage, { type ChatTurn } from "./components/ChatMessage";

const SAMPLES = [
  "기업가치는 얼마인가요?",
  "관리수수료의 주요 동인은 무엇인가요?",
  "AUM은 어떻게 추정되었나요?",
  "DCF에서 적용한 할인율(WACC)은?",
];

export default function App() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [dataset, setDataset] = useState("default"); // selected wiki version (?dataset=)
  const mainRef = useRef<HTMLElement>(null);
  const nextId = useRef(0);

  const scrollDown = useCallback(() => {
    requestAnimationFrame(() => {
      if (mainRef.current) mainRef.current.scrollTop = mainRef.current.scrollHeight;
    });
  }, []);

  useEffect(scrollDown, [turns, scrollDown]);

  // mutate the latest bot turn in place as stream events arrive
  const patchBot = (botId: number, fn: (t: ChatTurn) => ChatTurn) =>
    setTurns((ts) => ts.map((t) => (t.id === botId ? fn(t) : t)));

  const ask = useCallback(
    (question: string) => {
      if (busy || question.trim() === "") return;
      setBusy(true);

      const userId = nextId.current++;
      const botId = nextId.current++;
      setTurns((ts) => [
        ...ts,
        { id: userId, role: "user", text: question, trace: [], status: "done" },
        { id: botId, role: "bot", text: "", trace: [], status: "thinking" },
      ]);

      askStream(question, 8, {
        onStep: (s: TraceStep) =>
          patchBot(botId, (t) => ({ ...t, trace: [...t.trace, s] })),
        onAnswer: (answer) =>
          patchBot(botId, (t) => ({ ...t, text: answer || "(답변 없음)", status: "done" })),
        onError: (detail) =>
          patchBot(botId, (t) =>
            // only surface the error if no answer landed first
            t.status === "done" ? t : { ...t, text: detail, status: "error" },
          ),
        onDone: () => setBusy(false),
      }, dataset);
    },
    [busy, dataset],
  );

  const submit = () => {
    ask(input);
    setInput("");
  };

  return (
    <div className="app">
      <header>
        <div className="logo">S</div>
        <div>
          <h1>Project Stella</h1>
          <div className="sub">센트로이드 인베스트먼트 M&amp;A 밸류에이션 · 벡터리스 위키 에이전트</div>
        </div>
        <DatasetPicker value={dataset} onChange={setDataset} disabled={busy} />
        <StatusBadge />
      </header>

      <main ref={mainRef}>
        <div className="feed">
          {turns.length === 0 ? (
            <div className="empty">
              <h2>무엇이든 물어보세요</h2>
              <div>
                DCF, AUM, 관리수수료, 기업가치 등 — 답변은 셀 출처(<code>Sheet!Cell</code>)와 함께
                제공됩니다.
              </div>
              <div className="chips">
                {SAMPLES.map((q) => (
                  <button key={q} className="chip" onClick={() => ask(q)} disabled={busy}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((t) => <ChatMessage key={t.id} turn={t} />)
          )}
        </div>
      </main>

      <footer>
        <div className="footer-inner">
          <div className="hint">
            Enter 전송 · Shift+Enter 줄바꿈 · 추론 과정은 답변 아래에서 펼쳐볼 수 있습니다.
          </div>
          <Composer value={input} onChange={setInput} onSubmit={submit} busy={busy} />
        </div>
      </footer>
    </div>
  );
}
