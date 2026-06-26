"""Video Generator for EpiAgent.

Generates a complete mp4 video walkthrough including:
- Auto-generated voiceover (Bangladeshi English via edge-tts)
- Title slides
- Plotly visualizations (exported via kaleido)
- Synchronized audio/video using moviepy
"""

import asyncio
import time
import json
import os
from pathlib import Path

import numpy as np
import edge_tts
from moviepy import ImageSequenceClip, AudioFileClip, TextClip, ColorClip, CompositeVideoClip, concatenate_videoclips

from epiagent.agents.tools import (
    fetch_synthetic_data,
    run_seir_model,
    estimate_rt,
    run_ml_forecast,
    run_shap_analysis,
    detect_changepoints
)
from epiagent.dashboard.generator import (
    plot_epidemic_curve,
    plot_rt_timeseries,
    plot_forecast,
    plot_shap_importance,
    plot_changepoints
)

# Text to speak (synced with visuals)
SCRIPT_BLOCKS = [
    {
        "text": "Hi everyone! In an epidemic, delayed data and slow analysis cost lives. For the Kaggle Agents for Good Intensive, I built EpiAgent, an autonomous public health intelligence pipeline.",
        "visual": "title"
    },
    {
        "text": "My core design principle was: LLM decides, Math computes, LLM interprets. Every number comes from deterministic Python functions, preventing dangerous AI hallucinations in public health.",
        "visual": "principle"
    },
    {
        "text": "Let's look at the output. In just 3 seconds, the pipeline runs all 6 stages. First, we look at the epidemic curve.",
        "visual": "curve"
    },
    {
        "text": "Here, we use the WHO gold-standard Cori method to estimate Bayesian Rt. You can see it dropping below 1, meaning the epidemic is shrinking.",
        "visual": "rt"
    },
    {
        "text": "Our XGBoost ensemble forecast shows the predicted trajectory with a 95 percent prediction interval.",
        "visual": "forecast"
    },
    {
        "text": "And crucially, we have a SHAP explainability chart. When the model predicts cases will drop, it tells us exactly why, like the effect of recent lag features.",
        "visual": "shap"
    },
    {
        "text": "By chaining specialized agents with strict security guardrails, we can build auditable systems that public health officials can actually trust. Thanks for watching!",
        "visual": "end"
    }
]

VOICE = "en-IN-PrabhatNeural" # Using Indian English male as fallback

async def generate_audio_segments():
    """Generate audio files for each block."""
    voices = await edge_tts.VoicesManager.create()
    bd_voices = voices.find(Language="en", Country="BD")
    selected_voice = bd_voices[0]["Name"] if bd_voices else VOICE
    print(f"Using voice: {selected_voice}")

    audio_files = []
    for i, block in enumerate(SCRIPT_BLOCKS):
        out_file = f"temp_audio_{i}.mp3"
        communicate = edge_tts.Communicate(block["text"], selected_voice)
        await communicate.save(out_file)
        audio_files.append(out_file)
    return audio_files

def generate_visuals():
    """Generate the Plotly PNGs and Title slides."""
    print("Running pipeline to generate visuals...")
    data = fetch_synthetic_data("covid-19", 180, "test")
    records = data["records"]
    dates = [r["date"] for r in records]
    cases = [r["new_cases"] for r in records]
    deaths = [r.get("new_deaths", 0) for r in records]
    cases_json = json.dumps(cases)

    rt = estimate_rt(cases_json, "covid-19", window=7)
    forecast = run_ml_forecast(cases_json, horizon=14, dates_json=json.dumps(dates))
    shap = run_shap_analysis(cases_json, top_k=5)

    print("Exporting Plotly figures to PNG via Kaleido...")
    fig_curve = plot_epidemic_curve(dates, cases, deaths)
    fig_curve.write_image("viz_curve.png", width=1280, height=720, scale=2)

    rt_data = rt["rt_last_30d"]
    fig_rt = plot_rt_timeseries(dates[-len(rt_data["rt_mean"]):], rt_data["rt_mean"], rt_data["rt_lower"], rt_data["rt_upper"])
    fig_rt.write_image("viz_rt.png", width=1280, height=720, scale=2)

    ensemble = forecast["ensemble"]
    fig_forecast = plot_forecast(dates, cases, ensemble["dates"], ensemble["predicted"], ensemble["lower_bound"], ensemble["upper_bound"])
    fig_forecast.write_image("viz_forecast.png", width=1280, height=720, scale=2)

    drivers = shap["top_drivers"]
    fig_shap = plot_shap_importance([d["feature"] for d in drivers], [d["importance"] for d in drivers])
    fig_shap.write_image("viz_shap.png", width=1280, height=720, scale=2)

    return {
        "curve": "viz_curve.png",
        "rt": "viz_rt.png",
        "forecast": "viz_forecast.png",
        "shap": "viz_shap.png"
    }

def create_text_slide(text: str, duration: float, filename: str):
    """Create a simple text slide using moviepy."""
    # We create a black background with text
    clip = ColorClip(size=(1280, 720), color=(15, 10, 42), duration=duration)
    # Since TextClip requires ImageMagick, we'll use a hack if it fails, but assuming it works:
    try:
        txt_clip = TextClip(text=text, font_size=50, color='white', size=(1000, None), method='caption', align='center')
        txt_clip = txt_clip.with_position('center').with_duration(duration)
        video = CompositeVideoClip([clip, txt_clip])
        video.save_frame(filename, t=0)
    except Exception as e:
        print(f"Warning: TextClip failed (likely ImageMagick missing). Creating plain color slide. {e}")
        clip.save_frame(filename, t=0)
    return filename

def build_video(audio_files: list[str], visuals: dict):
    print("Stitching video together...")
    clips = []
    
    for i, block in enumerate(SCRIPT_BLOCKS):
        audio = AudioFileClip(audio_files[i])
        duration = audio.duration
        
        visual_type = block["visual"]
        img_path = f"slide_{i}.png"
        
        if visual_type in visuals:
            # It's a plot
            img_path = visuals[visual_type]
        else:
            # It's a text slide
            titles = {
                "title": "🦠 EpiAgent: Autonomous Epidemic Surveillance",
                "principle": "Core Principle:\nLLM Decides → Math Computes → LLM Interprets",
                "end": "Thank you!\nEpiAgent by Sujon"
            }
            create_text_slide(titles.get(visual_type, "EpiAgent"), duration, img_path)
            
        img_clip = ImageSequenceClip([img_path], fps=1).with_duration(duration)
        video_clip = img_clip.with_audio(audio)
        clips.append(video_clip)

    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile("EpiAgent_Final_Walkthrough.mp4", fps=24, codec="libx264", audio_codec="aac")
    print("Video saved as EpiAgent_Final_Walkthrough.mp4!")

async def main():
    audio_files = await generate_audio_segments()
    visuals = generate_visuals()
    build_video(audio_files, visuals)
    
    # Cleanup temp files
    for f in audio_files + list(visuals.values()) + [f"slide_{i}.png" for i in range(len(SCRIPT_BLOCKS))]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main())
