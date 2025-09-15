import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from typing import TypedDict
import json
import re
import os
from dotenv import load_dotenv

os.environ["GROQ_API_KEY"] = "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"
load_dotenv()

st.title("📝 Quiz Maker")

class GraphState(TypedDict):
    input: str
    summary: str
    quiz: str

def quiz_to_flashcards(quiz_json: str):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", quiz_json.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
    try:
        quiz_data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        st.error(f"❌ Could not parse quiz JSON: {e}")
        st.text(cleaned)
        return []
    return [{"front": q.get("question",""), "back": q.get("answer",""), "hint": q.get("explanation",""), "tag": q.get("type","")} for q in quiz_data]

def render_flashcards(flashcards):
    st.subheader("🃏 Flashcards")
    total = len(flashcards)
    flipped_count = sum(1 for i in range(total) if st.session_state.get(f"toggle_{i}", False))
    st.caption(f"Progress: {flipped_count}/{total} cards flipped")
    for i, card in enumerate(flashcards):
        with st.container():
            st.markdown(f"**Card {i+1}/{total}** — *{card['tag']}*")
            show = st.toggle("Show Answer", key=f"toggle_{i}")
            if show:
                st.markdown(f"**Back:** {card['back']}")
            else:
                st.markdown(f"**Front:** {card['front']}")
                if card["hint"]:
                    st.caption(f"💡 Hint: {card['hint']}")
        st.divider()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7, max_tokens=512)

summary_prompt = PromptTemplate.from_template("Summarize this text concisely in points format:\n{input_text}")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

quiz_prompt = PromptTemplate.from_template("Generate a JSON quiz from this text:\n{input_text}")
quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)

graph = StateGraph(GraphState)
graph.add_node("summarize", lambda s: {"summary": summary_chain.run(input_text=s["input"]).strip()})
graph.add_node("quiz", lambda s: {"quiz": quiz_chain.run(input_text=s["input"]).strip()})
graph.set_entry_point("summarize")
graph.add_edge("summarize", "quiz")
app = graph.compile()

with st.form("quiz_form"):
    user_input = st.text_area("Paste your text here:", height=200)
    submitted = st.form_submit_button("Generate Quiz & Summary")
    if submitted and user_input.strip():
        result = app.invoke({"input": user_input})
        flashcards = quiz_to_flashcards(result["quiz"])
        st.markdown("🧠 **Summary:**\n" + result["summary"])
        if flashcards:
            render_flashcards(flashcards)
