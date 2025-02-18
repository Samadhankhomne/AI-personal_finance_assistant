from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import date
from typing import Optional, Dict, List
import nest_asyncio
import fitz
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = OpenAIModel(api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

# transaction
class Transaction(BaseModel):
    date: date
    amount: float
    merchant: str
    category: Optional[str] = None

# Function to Extract Text from PDF
def Bank_Transaction_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text

# spending summary and recommendations
class SpendingSummary(BaseModel):
    total_expenses: float
    savings_recommendations: List[str]
    financial_goal_progress: Optional[Dict[str, float]] = Field(..., description="Progress toward financial goals as a percentage.")


# Define Categories
categories = ["Food", "Rent", "Entertainment", "Shopping", "Bills", "Transport", "Other"]


# Categorization Agent
categorization_agent = Agent(
    model=model,
    system_prompt="You are an AI categorization assistant."
                  " Given a transaction in the format 'Date: YYYY-MM-DD, Amount: , Merchant: XYZ',"
                  "classify it into all of the categories: ['Food', 'Rent', 'Entertainment', 'Shopping', 'Bills', 'Transport', 'Other']."
                  " Return all Bank Transaction and category name."
                  " Return all Bank Baleans"

                  f"pdf_path: {'Bank Transaction.pdf'}"
)

nest_asyncio.apply()

@categorization_agent.tool
def categoriz(ctx: RunContext[str], pdf_path: str): #-> Transaction:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text

# FINANCIAL ADVICE AGENT
advice_agent = Agent(
    model=model,
    result_type=SpendingSummary,
    deps_type=str,
    system_prompt="You are an AI financial advisor. "
                  "Given a breakdown of expenses, analyze spending patterns and suggest ways to save money. "
                  "Consider reducing non-essential spending and suggest budget-friendly habits.\n"
                  "**Instructions:**\n"
                  f"1. Extract transaction data from the provided PDF (pdf_path).\n"
                  "2. Categorize transactions using the provided categories (Spending).\n"
                  "3. Calculate the total expenses and category-wise breakdown.\n"
                  f"5. financial goal = 2000 "
                  "4. Provide at least 5 savings recommendations.\n"
                  "\n"
                  f"pdf_path: {'Bank Transaction.pdf'}\n"
                  f"Spending: {categories}"
    )

@advice_agent.tool
def Spending(ctx: RunContext[str], pdf_path: str) -> SpendingSummary:
  doc = fitz.open(pdf_path)
  text = "\n".join(page.get_text("text") for page in doc)
  return text

categorization_result = categorization_agent.run_sync(user_prompt=f"my transaction {categoriz}")
print(categorization_result.data)

sum_result = advice_agent.run_sync(user_prompt=f"{Spending}")
print(sum_result.data)