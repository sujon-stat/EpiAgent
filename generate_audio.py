import asyncio
import edge_tts
from pathlib import Path

# Extract just the spoken lines from the video script
TEXT_TO_SPEAK = """
Hi everyone! In an epidemic, delayed data and slow analysis cost lives. I wanted to see if AI agents could help solve this. For the Google Kaggle 'Agents for Good' Intensive, I built EpiAgent, an autonomous public health intelligence pipeline.

Early on, I realized a big problem. You can't just ask an AI to calculate a case fatality rate or a confidence interval. It might hallucinate the math, which is dangerous in epidemiology. So my core design principle was: LLM decides, Math computes, LLM interprets. Every number in this project comes from deterministic Python functions wrapped as tools.

I used Google's ADK to build a Sequential pipeline of 6 specialized agents: A Data Agent pulling surveillance records, a Security Agent that strips HIPAA identifiers, a Validator Agent running quality checks, an Analysis Agent running SEIR models and Bayesian estimation using the WHO gold-standard Cori method, an ML Agent forecasting with XGBoost, and a SitRep Agent synthesizing the final report.

Let's look at the output. In just 3 seconds, the pipeline runs all 6 stages and generates this interactive dashboard. You can see the Bayesian Rt dropping below 1 here, meaning the epidemic is shrinking. Our XGBoost forecast shows the predicted trajectory with a 95% interval. And crucially, we have a SHAP explainability chart. When the model predicts cases will drop, it tells us exactly why, like the effect of weekend reporting delays.

This project shows that AI agents in healthcare shouldn't just be massive monolithic chatbots. By chaining specialized agents with strict security guardrails and deterministic mathematical tools, we can build transparent, auditable systems that public health officials can actually trust. All 70 unit tests are passing, and the code is open source. Check out my Kaggle notebook linked below. Thanks for watching!
"""

# Try to use Bangladeshi English if available, otherwise Indian English (closest accent), otherwise standard English
VOICE = "en-IN-PrabhatNeural" # Indian English male as fallback

async def main():
    # Let's list voices to see if en-BD is available
    voices = await edge_tts.VoicesManager.create()
    bd_voices = voices.find(Language="en", Country="BD")
    in_voices = voices.find(Language="en", Country="IN")
    
    selected_voice = VOICE
    if bd_voices:
        selected_voice = bd_voices[0]["Name"]
        print(f"Found Bangladeshi voice: {selected_voice}")
    elif in_voices:
        # Filter for a male voice if possible
        male_in = [v for v in in_voices if v["Gender"] == "Male"]
        selected_voice = male_in[0]["Name"] if male_in else in_voices[0]["Name"]
        print(f"Using Indian English voice: {selected_voice}")
    
    print(f"Generating audio with voice: {selected_voice}...")
    communicate = edge_tts.Communicate(TEXT_TO_SPEAK, selected_voice)
    await communicate.save("epiagent_video_voiceover.mp3")
    print("Audio successfully saved to epiagent_video_voiceover.mp3")

if __name__ == "__main__":
    asyncio.run(main())
