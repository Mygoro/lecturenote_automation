# Lecture Automation Pipeline — 전체 설계 문서

> **이 파일의 용도**: Claude Code가 매 세션마다 읽는 ground truth.
> 설계 변경이 있으면 이 파일을 먼저 수정하고 작업에 들어가라.
> 지시 예시: "PIPELINE_PLAN.md 읽고 Phase 2 Step 2 실행해라"
> 작업을 완료했다면 각 작업 내용을 검증하는 방법을 안내하고 검증까지 완료하면 이 파일에 작업 진행 현황을 업데이트하라.
---

## 프로젝트 개요

강의 녹음 → 자동 전사 → 구조화 요약 → Notion 업로드 파이프라인.
개인 사용 목적. 시험공부 시 이것만 봐도 충분한 원스탑 솔루션이 목표.

**현재 코드 위치**: `C:\Users\win10\Claude Series\lecture_notes\automation\`

---

## 전체 단계 정의

| 단계 | 내용 | 상태 |
|------|------|------|
| Phase -1 | 보안 + 환경 정리 | ✅ 완료 |
| Phase 2 | 요약 파이프라인 품질 개선 | ✅ 완료 |
| Phase 3 | 클라우드 이전 + 자동화 안정화 | ✅ 완료 |
| Phase 4 | 동적 라우팅 + 에러 알림 | 🔄 진행 중 |
| Phase 5 | PPT/PDF 융합 | ⏳ 대기 |
| Phase 6 | 퀴즈 생성 | ⏳ 대기 |
| Phase 7 | UI | ⏳ 대기 |

---

## Phase -1 완료 내역 (참고용)

- `.env` OneDrive 밖으로 이동 완료
- API 키 전체 교체 (Anthropic, OpenAI) + spend limit 설정 ($50/월)
- `.gitignore` 생성 (`.env`, `token.json`, `google_credentials.json`, `processed_files.json`, `__pycache__`, `*.pyc` 제외)
- Git 저장소 초기화 + 첫 커밋
- `__pycache__` cpython-314 파일 삭제
- `register_scheduler.bat` Python 3.10 (`C:\Python310\python.exe`)으로 고정

---

## Phase 2 상세 설계

### 목표

현재 `process_transcript()`가 풀스크립트를 Sonnet에 통째로 넘기는 구조를
다단계 파이프라인으로 교체해 요약 품질을 높인다.

**핵심 변경 3가지**:
1. Whisper를 `verbose_json`으로 변경해 word-level timestamp 확보
2. 청크 분할 → 분류 → 구조 추출 → 토픽별 요약 순서로 단계화
3. Sonnet 출력을 구조화 JSON으로 강제 (tool use), temperature=0.2 고정

---

### 현재 파이프라인 vs 변경 후

**현재**:
```
transcribe_audio()        ← response_format="text", language="en" 고정
  → process_transcript()  ← 풀스크립트 통째로 Sonnet 1회 호출
      ├─ _clean_transcript()
      ├─ extract_notices() ← Haiku 별도 호출 (중복 입력)
      └─ summarize()       ← temperature 미설정(1.0), 구조화 없음
  → upload_to_notion()
```

**변경 후**:
```
transcribe_audio()         ← verbose_json, language=None(자동감지)
  → segment_transcript()   ← NEW: 무음+길이 기반 청크 분할
  → classify_chunks()      ← NEW: Haiku로 각 청크 라벨링 + 공지 추출
  → extract_structure()    ← NEW: Sonnet으로 토픽 추출 (본론 청크만)
  → build_summary()        ← NEW: 토픽별 요약 + 출처 인용 강제
  → upload_to_notion()     ← 기존 유지, 입력 구조만 변경
```

---

### 변경 후 파일 구조

```
automation/
  transcribe.py           ← 기존 유지, verbose_json + language 파라미터만 수정
  summarize.py            ← pipeline/ 함수들을 순서대로 호출하는 진입점으로 단순화
  notion_uploader.py      ← 기존 유지, 새 스키마 입력 받도록 수정
  main.py                 ← 기존 유지
  pipeline/
    __init__.py
    schema.py             ← dataclass로 전체 스키마 정의
    segmenter.py          ← segment_transcript()
    classifier.py         ← classify_chunks()
    structurer.py         ← extract_structure()
    summarizer.py         ← build_summary()
```

---

### 함수 시그니처 및 데이터 흐름

#### 1. `transcribe.py` 수정 (Step 1)

변경 사항:
- `response_format`: `"text"` → `"verbose_json"`
- `language` 기본값: `"en"` → `None` (None이면 API에 language 파라미터 미전송)
- 반환값: `(text, duration_minutes)` → `(verbose_json_result, duration_minutes)`

```python
def transcribe_audio(api_key: str, file_path: str, language: str = None) 
    -> tuple[dict, float]:
    # verbose_json_result: Whisper API 반환 전체 객체
    # duration_minutes: 음성 길이 (비용 계산용, 기존과 동일)
```

---

#### 2. `pipeline/schema.py` (Step 2)

전체 파이프라인에서 사용하는 dataclass 정의.

```python
@dataclass
class Segment:
    index: int
    start: str        # "00:00:00"
    end: str          # "00:12:34"
    text: str

@dataclass
class LabeledSegment:
    index: int
    start: str
    end: str
    text: str
    label: str        # "lecture_core" | "announcement" | "qa" | "example" | "smalltalk"
    notices: list     # label=="announcement"일 때만 채워짐

@dataclass
class Topic:
    title: str
    order: int
    source_chunks: list[int]
    time_range: str   # "00:05:00-00:18:30"

@dataclass
class KeyPoint:
    claim: str
    explanation: str
    source_quote: str
    timestamp: str

@dataclass
class TopicSummary:
    title: str
    order: int
    time_range: str
    key_points: list[KeyPoint]
    important_emphasis: list[str]
    concepts_introduced: list[str]

@dataclass
class Notice:
    type: str         # "assignment" | "exam" | "attendance" | "schedule_change" | "notice"
    content: str
    deadline: str | None
    source_quote: str
    timestamp: str

@dataclass
class QASegment:
    question: str
    answer: str
    timestamp: str

@dataclass
class SummaryResult:
    lecture_meta: dict        # course, week, duration, language
    announcements: list[Notice]
    topics: list[TopicSummary]
    qa_segments: list[QASegment]
    full_transcript: str
```

---

#### 3. `pipeline/segmenter.py` (Step 3)

```python
def segment_transcript(verbose_json: dict) -> list[Segment]:
```

로직:
- Whisper verbose_json의 `segments` 배열에서 각 세그먼트의 start/end/text 사용
- 연속된 세그먼트 간 gap이 3초 이상이면 청크 경계로 인식
- 청크 길이 5분 미만이면 다음 청크와 합침
- 청크 길이 15분 초과이면 강제 분할

---

#### 4. `pipeline/classifier.py` (Step 4)

```python
def classify_chunks(segments: list[Segment], api_key: str) -> list[LabeledSegment]:
```

- 모델: `claude-haiku-4-5-20251001`
- 각 청크를 개별 Haiku 호출로 라벨링
- 라벨: `lecture_core` / `announcement` / `qa` / `example` / `smalltalk`
- `announcement` 라벨인 경우 notices 필드도 함께 추출 (기존 Haiku 공지 추출 통합)
- 출력: JSON 강제 (tool use)

프롬프트 핵심:
```
Classify this lecture audio segment into one of these labels:
- lecture_core: main lecture content
- announcement: assignments, exams, schedule changes, attendance
- qa: student questions and professor answers
- example: examples or case studies
- smalltalk: greetings, off-topic, filler

If label is "announcement", also extract notices as a list.
Each notice: {type, content, deadline}

Return JSON only.
```

---

#### 5. `pipeline/structurer.py` (Step 5)

```python
def extract_structure(labeled_segments: list[LabeledSegment], api_key: str) -> list[Topic]:
```

- `lecture_core` 라벨 청크만 필터링해서 Sonnet에 전달
- "이 청크들에서 주요 토픽 N개 추출, 각 토픽이 어느 청크 index에 근거하는지 매핑"
- 출력: JSON 강제 (tool use)
- 모델: `claude-sonnet-4-6`

---

#### 6. `pipeline/summarizer.py` (Step 6)

```python
def build_summary(
    topics: list[Topic],
    labeled_segments: list[LabeledSegment],
    lecture_meta: dict,
    api_key: str
) -> SummaryResult:
```

- 토픽별로 Sonnet 호출 (토픽에 해당하는 청크 텍스트만 입력)
- temperature=0.2 고정, max_tokens=4096 (stop_reason=="max_tokens"시 8192로 즉시 재시도 1회)
- 출력: tool use로 JSON 강제

프롬프트 핵심 제약:
```
For each key point:
- state the claim
- provide a concise explanation
- quote the professor's exact words (source_quote)
- include the timestamp

For important_emphasis: detect phrases like 
"this is important", "key point", "remember", "critical", 
"make sure you understand", or any repeated emphasis.

Do not invent content. Every claim must be traceable to the transcript.
```

---

### 최종 출력 스키마 (Notion 업로드 입력)

```json
{
  "lecture_meta": {
    "course": "AI Agents",
    "week": 7,
    "duration": "01:23:45",
    "language": "en"
  },
  "announcements": [
    {
      "type": "exam",
      "content": "중간고사 다음 주 목요일",
      "deadline": "Week 8",
      "source_quote": "So the midterm is going to be next Thursday",
      "timestamp": "00:02:15"
    }
  ],
  "topics": [
    {
      "title": "Reward Function Design",
      "order": 1,
      "time_range": "00:05:00-00:18:30",
      "key_points": [
        {
          "claim": "Sparse rewards make learning difficult",
          "explanation": "보상이 드물게 주어지면 에이전트가 어떤 행동이 좋은지 파악하기 어렵다",
          "source_quote": "When you have a sparse reward, the agent just doesn't know what it did right",
          "timestamp": "00:08:42"
        }
      ],
      "important_emphasis": [
        "sparse reward is the key challenge in real-world RL"
      ],
      "concepts_introduced": ["sparse reward", "reward shaping"]
    }
  ],
  "qa_segments": [
    {
      "question": "학생 질문 내용",
      "answer": "교수 답변 내용",
      "timestamp": "00:45:10"
    }
  ],
  "full_transcript": "전체 텍스트..."
}
```

---

## Claude Code 운용 규칙

1. **한 Step씩 실행**한다. 완료 후 반드시 보고하고 멈춰라.
2. **확인 전 다음 Step 금지**. "Step N 완료, 다음으로 넘어갈까요?" 후 대기.
3. **코드 작성 전 설명 먼저**. 뭘 어떻게 바꿀지 설명 후 구현.
4. **다른 파일 건드리지 마라**. 해당 Step에서 명시한 파일만 수정.
5. **실행은 지시가 있을 때만**. 코드 작성 후 "실행할까요?" 물어봐라.
6. **이 파일(PIPELINE_PLAN.md)이 ground truth**. 설계가 모호하면 여기서 확인.

---

## 각 Step 완료 기준

| Step | 작업 | 완료 기준 |
|------|------|-----------|
| Step 1 | `transcribe.py` 수정 | ✅ 완료 |
| Step 2 | `pipeline/schema.py` 생성 | ✅ 완료 |
| Step 3 | `pipeline/segmenter.py` 생성 | ✅ 완료 |
| Step 4 | `pipeline/classifier.py` 생성 | ✅ 완료 |
| Step 5 | `pipeline/structurer.py` 생성 | ✅ 완료 |
| Step 6 | `pipeline/summarizer.py` 생성 | ✅ 완료 |
| Step 7 | `summarize.py` 리팩토링 | ✅ 완료 |
| Step 8 | `notion_uploader.py` 수정 | ✅ 완료 |
| Step 9 | `main.py` 수정 | ⏳ 대기 — 전체 파이프라인 end-to-end 실제 파일로 테스트 통과 |

---

## Phase 3 이후 메모 (지금 작업 안 함)

- **Phase 3**: GitHub Actions + Drive 폴링으로 클라우드 이전. Windows 작업스케줄러 제거.
- **Phase 4**: 동적 폴더 라우팅 (`.lecture-config.json` 메타파일 방식). 에러 시 텔레그램/Discord 알림.
- **Phase 5**: PPT/PDF 융합. `python-pptx`로 슬라이드 텍스트 추출 + 임베딩 매칭으로 음성-슬라이드 정렬.
- **Phase 6**: 토픽별 퀴즈 자동 생성. 정답이 본문에 있는지 자체 검증 포함.
- **Phase 7**: UI. Next.js + Vercel 또는 Notion 자체 UI. 백엔드 안정 후 착수.
- **Whisper 로컬 이전**: 데스크탑 GPU 활용 가능. Phase 3 완료 후 한 학기 비용 보고 결정.
