import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from typing import TypedDict
import os
from dotenv import load_dotenv

os.environ["GROQ_API_KEY"] = "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"
load_dotenv()

st.title("😃 Mood Detection")

class GraphState(TypedDict):
    input: str
    mood: str
    output: str

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7, max_tokens=512)

# --- Nodes ---
mood_prompt = PromptTemplate.from_template(
    "Classify the mood of this message: {input}. Choose one: foggy, focused, frustrated, angry, happy."
)
mood_chain = LLMChain(llm=llm, prompt=mood_prompt)

response_prompt = PromptTemplate.from_template(
    "The user is feeling {mood}. Respond helpfully to their message: {input}"
)
response_chain = LLMChain(llm=llm, prompt=response_prompt)

def detect_mood(state: GraphState) -> GraphState:
    mood = mood_chain.run(input=state["input"]).strip().lower()
    return {**state, "mood": mood}

def generate_response(state: GraphState) -> GraphState:
    output = response_chain.run(input=state["input"], mood=state["mood"])
    return {**state, "output": output}

graph = StateGraph(GraphState)
graph.add_node("detect_mood", detect_mood)
graph.add_node("respond", generate_response)
graph.set_entry_point("detect_mood")
graph.add_edge("detect_mood", "respond")
app = graph.compile()

with st.form("mood_form"):
    user_input = st.text_input("Type something:")
    submitted = st.form_submit_button("Analyze Mood")
    if submitted and user_input.strip():
        result = app.invoke({"input": user_input, "mood": "", "output": ""})
        st.write("🧠 Mood:", result["mood"])
        st.write("💬 Response:", result["output"])
