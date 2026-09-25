from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List



class QuizQuestion(BaseModel):
    question_text: str = Field(
        description="Simple, Grade-2 appropriate question in Kenyan context"
    )
    options: List[str] = Field(
        description="List of 4 answer options",
        min_items=4,
        max_items=4
    )
    correct_answer: str = Field(description="The exact correct string from the options")
    hint: str = Field(description="A helpful, encouraging hint if the child gets stuck")


class QuizResponse(BaseModel):
    competency_band: str
    strand: str
    questions: List[QuizQuestion]
    teacher_insight_summary: str = Field(
        description="Diagnostic summary mapped to KICD standards"
    )
