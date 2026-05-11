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
    type: str           # "assignment" | "exam" | "attendance" | "schedule_change" | "notice"
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
    lecture_meta: dict
    announcements: list[Notice]
    topics: list[TopicSummary]
    qa_segments: list[QASegment]
    full_transcript: str
