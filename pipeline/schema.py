from dataclasses import dataclass, field


@dataclass
class Segment:
    index: int
    start: float    # 초 단위 (Whisper 반환값 그대로)
    end: float      # 초 단위 (Whisper 반환값 그대로)
    text: str


@dataclass
class LabeledSegment:
    index: int
    start: str
    end: str
    text: str
    label: str      # "lecture_core" | "announcement" | "qa" | "example" | "smalltalk"
    notices: list = field(default_factory=list)  # label=="announcement"일 때만 채워짐


@dataclass
class Topic:
    title: str
    order: int
    source_chunks: list[int]
    time_range: str     # "00:05:00-00:18:30"


@dataclass
class Structure:
    """structurer 1회 호출로 얻는 전체 구조.

    토픽 추출과 함께 (a) 학생이 해야 할 일, (b) 어느 토픽에도 담기지 않은 사실을
    같이 뽑는다. structurer만이 전체 세그먼트를 한 번에 보므로, 여기서 잡지 않으면
    토픽 단위로 동작하는 summarizer는 '애초에 안 들어온 내용'을 볼 수 없다.
    """
    topics: list[Topic]
    action_items: list[str] = field(default_factory=list)
    uncovered_points: list[str] = field(default_factory=list)


@dataclass
class KeyPoint:
    claim: str
    explanation: str
    source_quote: str   # 전사문 원문 그대로. 대조 실패 시 빈 문자열로 비운다.


@dataclass
class TopicSummary:
    title: str
    order: int
    time_range: str
    key_points: list[KeyPoint]
    concepts_introduced: list[str]
    summary_oneliner: str = ""
    keywords_with_brief: list[dict] = field(default_factory=list)


@dataclass
class Notice:
    type: str           # "assignment" | "exam" | "attendance" | "grading" | "logistics" | "schedule_change" | "notice"
    content: str
    deadline: str | None
    source_quote: str   # 전사문 원문 그대로. 대조 실패 시 빈 문자열로 비운다.


@dataclass
class QASegment:
    question: str
    answer: str


@dataclass
class SummaryResult:
    lecture_meta: dict
    announcements: list[Notice]
    topics: list[TopicSummary]
    qa_segments: list[QASegment]
    full_transcript: str
    action_items: list[str] = field(default_factory=list)
    uncovered_points: list[str] = field(default_factory=list)
    glossary: list[dict] = field(default_factory=list)   # [{"keyword","brief"}] 중복 제거된 통합 용어집
