# MA — 센트로이드 M&A 가치평가 지식베이스 + 질의 에이전트

**센트로이드인베스트먼트파트너스**(Centroid Investment Partners)와 그 운용사 **센트로이드매니지먼트**의
M&A 가치를 추정하기 위한 프로젝트입니다. 원천(source of truth)은 63개 시트의 엑셀 밸류에이션
모델이며, 이 워크북을 **질의 가능한 지식베이스**로 끌어올리고 그 앞에 **출처를 인용하는 에이전트**를
둡니다. 그래서 *"무엇이 DCF를 움직이나?"*, *"수수료/AUM 가정이 기업가치까지 어떻게 흘러가나?"*,
*"리포트는 인건비가 영업비용의 53.9%라는데 맞나?"* 같은 질문에 **결정론적이고, 셀 단위로 인용되며,
감사 가능한** 답을 냅니다.

> 설계 상세, 워크북 구조, 프로퍼티 그래프 스키마는 [`CLAUDE.md`](./CLAUDE.md)에 있습니다.
> 이 README는 실무용 개요 + 실행 가이드 + **작업 로그**입니다.

## 접근 방식: 기본은 vectorless(벡터 DB 없음)

데이터가 구조적이고(수식 의존성 그래프), 숫자·셀참조는 임베딩이 잘 안 되며, M&A 가치평가는
top-k 벡터 검색이 보장할 수 없는 **완전하고 감사 가능한 provenance**가 필요합니다. 그래서 **벡터
DB가 없습니다.** 검색은 그래프 순회와 결정론적 위키 조회이고, LLM은 *단어 → 노드*(질의어를
노드로 해석)와 최종 자연어 합성에만 쓰며 **근거 수집에는 절대 쓰지 않습니다.**

같은 워크북에서 두 가지 KB 패러다임을 빌드합니다.

| 패러다임 | 내용 | 코드 |
|---|---|---|
| **Graph(그래프)** | 셀 단위 수식 DAG(`DEPENDS_ON`)를 의미 프로퍼티 그래프(Entity/Fund/Metric/Period…)로 lift | `src/stella_kb/graph/` |
| **Wiki(위키)** | 시트당 마크다운 페이지 1장(결정론적 사실 + LLM 산문), `INDEX.md` 목차, 원장 사이드카 | `src/stella_kb/wiki/` |

## 질의 에이전트

위키를 탐색해 한국어로 셀을 인용하며 답하는 LangGraph 파이프라인(`apps/agent/`)입니다.

```
planner → (fan-out) solve×N → auditor → synthesizer
            │                    │           │
   질문을    router→retriever→   교차근거    셀 인용
   하위질문   verifier 루프로     감사       한국어 답변
   으로 분해  위키페이지+원장 읽기
```

- **solve** 브랜치는 하위질문마다 하나씩 동시 실행되며, 각자 결정론적 위키 읽기를 수행합니다.
- **auditor**는 규칙 기반 교차근거 검사(동일 셀 중복인용·PDF주장 vs 원천·미응답 하위질문)를 돌려
  synthesizer가 반드시 반영해야 할 경고(caveat)를 냅니다.
- **synthesizer**는 evidence + 경고를 바탕으로 보수적으로 답합니다. 산술은 모든 입력이 evidence에
  있을 때만 직접 수행하고, 출처 셀을 함께 인용하며, 불확실하면 "확인 불가"로 명시합니다.
- 별도의 **DART 백엔드**는 *국내 상장사* 질문을 DART 공시 MCP 서버로 답하며, 라우터가 질문을
  wiki/dart로 분류합니다.

모든 프롬프트 파일은 한국어로 작성되어 있습니다(JSON 키·셀참조는 원형 유지).

## 레이아웃

```
data/                 # 버전별 빌드 산출물 (gitignore). 각 버전은 data/<version>/ 아래 자기완결
  v0.1/  { raw md parsed wiki }   # 정본 63시트 센트로이드 모델 — 기본(default) 데이터셋
  v0.2/  { raw md parsed wiki }   # 멀티덱 비전 테스트셋: 센트로이드 원장 + CAESAR/LIFE/STELLA FDD 덱
  eval/  graph/  logs/            # 평가 출력 · 그래프 산출물 · 로그
src/stella_kb/        # KB 빌드 파이프라인
  graph/              # 수식 DAG → 의미 프로퍼티 그래프 (extract / semantic / metrics / lift / query)
  wiki/               # 워크북 → 마크다운 위키 (dump_md → parse_llm → compile → index → pdf_pages) + ledger
    pdf_pages.py      # PDF 적재: 비전 → 구조화 figure + 원문 표/[다이어그램] 엣지목록 · 덱별 2층 문서노드
  parsers/pdf/        # 비전 기반 PDF 파서 (표·[그래프]·[다이어그램] 추출)
  llm.py  config.py   # OpenAI 호환 vLLM 클라이언트 · 중앙 설정(env > config.yaml > default; 경로 접근자)
apps/agent/           # 질의 에이전트
  core.py             # 공개 API/파사드: run / ask / answer(라우터) / stream_run — backends/ 백엔드로 분기
  datasets.py         # 데이터셋(위키 버전) 레지스트리 + 캐시 WikiStore (id → 위키 디렉터리)
  backends/             # 에이전트 백엔드 (core가 여기로 분기)
    supervisor.py     #   supervisor StateGraph: wiki/dart 워커 라우팅·병합, 스트리밍 fast-path
    dart.py           #   DART(상장사) tool-calling 백엔드
    wiki/             #   wiki LangGraph: state · nodes(planner/solve/auditor/synthesizer) · build
  retrieval/          # 결정론적 위키 접근 (lookup, open_page, trace_links, query_ledger)
  api/                # FastAPI: /ask(GET) · /ask/stream(GET SSE) · /datasets · /health
  prompts/            # 한국어 프롬프트 (route, planner, router, retriever, verifier, synthesizer, supervisor)
frontend/             # React + Vite 채팅 UI (SSE) · DatasetPicker(버전 선택) · API 프록시
mcps/dart-mcp/        # 벤더링한 DART 공시 MCP 서버 (Docker, SSE :8002, 토큰 인증)
eval/                 # stella_crosscheck(v0.1 tier) · qa_eval(v0.2 비전-QA rubric) · ragas_eval
scripts/              # run_server.sh, run_pipeline.sh, run_eval.sh, run_qa_eval.sh, serve_*.sh
config.yaml           # 중앙 설정 (env > yaml > default) · agent.datasets 버전 레지스트리; 비밀값은 .env에만
```

## 설치

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

에이전트는 로컬 OpenAI 호환 **vLLM**(gemma-4-31B-it, `http://123.37.5.219:8001/v1`)을 호출합니다 —
`scripts/serve_gemma.sh` 참고. 점검: `curl -s 123.37.5.219:8001/v1/models`.

## 지식베이스 빌드

```bash
# 그래프 패러다임
python -m src.stella_kb.graph.extract     # 수식 → 약 13.7k 셀, 약 74k DEPENDS_ON 엣지
python -m src.stella_kb.graph.metrics     # 큐레이션 cell→Metric 앵커 → 102 metric, 14 period
python -m src.stella_kb.graph.semantic    # 전체 의미 그래프 → data/graph/stella_graph.json
python -m src.stella_kb.graph.query       # 질의: resolve → traverse → 인용 답변

# 위키 패러다임 — 기본은 data/v0.1/ 에 빌드. 새 버전은 env로 디렉터리만 지정(코드 수정 불필요):
scripts/run_pipeline.sh                                   # 정본(v0.1) 빌드 → data/v0.1/wiki/
MNA_WIKI_WORKBOOK=<x.xlsx> MNA_WIKI_DATA=data/v0.2 \
  MNA_WIKI_PDF_DIR=test_data/v0.2 scripts/run_pipeline.sh # 새 버전 빌드 → data/v0.2/wiki/
# 빌드 후 config.yaml 의 agent.datasets 에 한 줄 등록하면 API/UI에서 선택 가능.
```

(`data/`의 빌드 산출물은 재생성 가능하며 gitignore 대상 — `src/`만 커밋, `data/`는 커밋 금지.)

### 데이터셋(버전) 선택

에이전트는 등록된 데이터셋을 **요청 단위로** 선택해 답합니다(동시성 안전). `config.yaml`의
`agent.datasets`가 안전한 id → 위키 디렉터리를 매핑하고, 요청에 `dataset` 파라미터로 고릅니다.

```bash
curl "localhost:5001/datasets"                              # 등록·빌드된 버전 목록
curl "localhost:5001/ask?question=기업가치는?&dataset=v0.2"  # 특정 버전 질의 (GET+Query)
```

## 에이전트 실행 (API + UI)

```bash
scripts/run_server.sh                      # FastAPI :5001 (mcps/dart-mcp/.env에서 DART 토큰 로드)
cd frontend && npm install && \
  VITE_API_TARGET=http://localhost:5001 npm run dev   # React UI :5173, API 프록시
```

사용 포트: **:5001** 에이전트 API · **:5173** 프런트엔드 · **:8001** vLLM · **:8002** DART MCP.
터미널을 닫아도 유지하려면 `setsid … &`로 분리 실행(로그는 `.run/` 아래).

### DART 백엔드

`mcps/dart-mcp/`는 벤더링한 DART 공시 MCP 서버(Docker, SSE :8002, bearer 토큰 인증)입니다.
에이전트는 `DART_MCP_TOKEN`으로 인증하며 `run_server.sh`가 `mcps/dart-mcp/.env`에서 이를 로드합니다.
비밀값(`DART_API_KEY`, `DART_MCP_TOKEN`)은 `.env`에만 두고 소스에는 절대 넣지 않습니다.

## 평가

`test_data/` 아래 두 정답셋을 공유 vLLM으로 채점하고 결과는 `data/eval/`에 씁니다.

```bash
scripts/run_qa_eval.sh                      # v0.2 비전-QA 54문항(덱 차트/구조도/매트릭스), rubric 채점
                                            #   → data/eval/v0.2 (doc·capability C1~C5·visual_type 분해)
scripts/run_eval.sh                         # v0.1 3-tier PDF×Excel 교차검증 20문항 (지정 위키 대상)
.venv-ragas/bin/python -m eval.ragas_eval   # RAGAS: grounded_faithfulness, answer_correctness, …
```

> ⚠️ **평가는 노이즈가 큽니다.** 공유 vLLM은 temperature 0에서도 비결정적(연속 배칭)이고, 위키
> 재빌드는 페이지별 `structure_section`을 다시 실행하므로, 단일 런의 ±0.1 미만 점수 차이는 신호가
> 아닙니다. **여러 런의 평균**으로 비교하고, 에이전트만 바꾸는 A/B에서는 빌드된 페이지를 고정하세요.

**현재 베이스라인(tier 판정): T1 0.96 · T2 0.80 · T3 1.00 · 전체 0.93** (20문항). 스톡 RAGAS의
faithfulness/correctness는 결정론적-산술 답변에서 오보정되므로, 골든/산술 인지형 커스텀
`DiscreteMetric`을 사용합니다. (테스트 데이터는 기밀이며 gitignore 대상.)

## 테스트

```bash
pytest                 # 결정론적·오프라인 스위트 (~2s, 네트워크 없음)
pytest --run-llm       # 라이브 vLLM end-to-end 스모크 테스트까지 실행
```

## 작업 로그

최신순. 전체 이력은 `git log` 참고.

- **데이터 버전 관리 + `data/` 재구성.** 각 코퍼스를 `data/<version>/`(raw/md/parsed/wiki)로
  자기완결화. 정본 = `data/v0.1`, 신규 멀티덱 테스트셋 = `data/v0.2`. 평가 출력은 `data/eval/`,
  그래프 산출물은 `data/graph/`, 로그는 `data/logs/`. 모든 경로는 `config.py` 접근자로 해석
  (하드코딩 제거) — 빌드/서빙/평가는 env로 디렉터리만 지정.
- **데이터셋(버전) API + 프런트 선택기.** `apps/agent/datasets.py` 레지스트리(`config.yaml`
  `agent.datasets`) + 캐시 `WikiStore`. `/ask`·`/ask/stream`에 `dataset` 파라미터(둘 다 GET+Query;
  `/ask` POST 제거), `/datasets` 엔드포인트, React `DatasetPicker`. 위키 디렉터리를 글로벌이 아닌
  요청별 인자로 그래프에 스레딩 → **동시성 안전**.
- **v0.2 비전-QA 디버깅 (0.22 → ~0.75).** 54문항 비전 정답셋(`eval/qa_eval.py`)으로 진단·수정:
  (1) **에이전트 dedup 버그** — 모든 PDF 행이 같은 `[FDD<n>]` 태그라 `(page,cell)` 키가 시계열
  전체를 한 행으로 붕괴 → 키를 `(page,cell,period,term)`로; auditor의 PDF-only/중복셀 경고 게이팅 +
  synthesizer가 리포트 수치로 답하도록(확인불가 남발 제거). **0.22 → 0.54**. (2) **적재 수정** —
  비전이 충실히 옮긴 표/매트릭스를 structurer가 떨어뜨리던 것을 **원문 그리드 그대로** 페이지에
  싣고, figure 없는 페이지도 보존(드롭된 STELLA 페이지 복구), 표 헤더·행라벨을 alias로 → 라우팅
  개선. **→ ~0.70**. (3) **구조도 인식** — 비전 프롬프트가 조직도/지배구조도를 `[다이어그램]`
  엣지목록(출발→도착 : %·금액 + 색 범례)으로 추출, "페이지 N" 무의미 제목을 LLM title로 대체. (4)
  **덱별 2층 인덱스** — PDF당 상세 설명(상층) + 페이지 ToC(하층). ⚠️ 공유 vLLM 비결정성으로 단일
  런 비교는 노이즈가 큼(평균으로 비교).
- **계산(compute) 노드 — 실험 후 리버트.** auditor와 synthesizer 사이에 결정론적 산술 노드를
  넣어 LLM은 산술 *식*만 제안하고 안전한 AST 계산기가 평가하도록 시도. 단위 테스트·프로덕션
  단발 질의(인건비/영업비용 비율)에서는 정확히 동작했으나, **교차검증 평가에서 회귀**: 전체
  0.93 → 0.85, 특히 T3 검증 불가에서 1.00 → 0.67. "단정 불가"여야 하는데도 확신 있는 숫자를
  만들어냄(예: Q20이 PDF 출처 501값을 이질적 출처 단순합으로 재현했다고 단정 — T3의 치명적
  오류). 경고-종속 + 절제 프롬프트 가드로 T3 1.00·전체 0.90까지 회복했으나 여전히 0.93
  베이스라인 미만이라 **리버트**. 평가셋이 절제 위주(20문항 중 8개 T2·T3)라 정확 산술의 이득이
  작고, 보수적 synthesizer가 더 잘 보정됨. `6bffcca`(revert of `1dfc8a7`)
- **DART 백엔드 강화** — `run_server.sh`가 `DART_MCP_TOKEN`을 로드해 MCP 서버 인증을 가능하게
  하고, 라우터를 강화해 재무/리포트 어휘만으로 내부 질문이 DART로 오라우팅되지 않게 함(상장사
  회사명이 있을 때만 DART). `45a5571`
- **`mcps/dart-mcp` 벤더링** — DART 공시 MCP 서버(업스트림 `2geonhyup/dart-mcp`)에 로컬 수정 +
  Docker SSE 서빙을 더해 저장소에 포함. `13d661c`
- **거래내역 원장 쿼리** — 원장 사이드카에서 거래내역(시계열 파싱이 떨어뜨리는 행)을 결정론적으로
  필터·합산해 거래 질문의 검색 공백을 메움. `7fd9c9f`
- **summary 기반 라우팅 + FDD→Excel 교차참조** — PageIndex 영감: 라우터가 각 페이지의 한 줄
  요약을 읽고, FDD 페이지가 펀드 식별자로 Excel 원천에 링크. `fdcc1c3`
- **비전 PDF 파서** — FDD 적재를 비전 전용으로(gemma는 멀티모달); breadcrumb 추출로 페이지
  라벨 정리. `eb4a285`, `9cda824`, `70dbe6c`
- **설정 중앙화** — `config.yaml` + `src/stella_kb/config.py`(env > yaml > default); 비밀값은
  `.env`에. `d520530`
- **RAGAS 평가** — 비동기 하니스 + 산술 공정형 커스텀 DiscreteMetric(스톡 지표는 파생 숫자
  답변에서 false-negative). `cdc9a36`, `70dbe6c`

## 참고

- 패턴을 차용한(의존성이 아닌) **참조 설계**: OpenKB(한 번 컴파일, 화이트리스트 가드 `[[wikilinks]]`),
  DCI-Agent-Lite(질의시 search→inspect→verify 루프), PageIndex(vectorless 트리 라우팅).
- **캐시값 주의**: openpyxl은 재계산하지 않으며, 엑셀이 재계산하지 않은 셀은 `None`으로 읽힘.
  최신 숫자는 Excel/LibreOffice에서 재계산 후 추출.
- 기밀 입력(`data/`의 워크북, `test_data/`의 FDD)은 **gitignore 대상**이며 절대 커밋 금지.
