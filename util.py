import json
import requests

def load_lottie():
    """Load animations (example URLs or local files)."""
    lottie_url1 = "https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json"
    lottie_url2 = "https://assets4.lottiefiles.com/packages/lf20_tll0j4bb.json"
    return lottie_url1, lottie_url2

def stream_data(text):
    """Stream text gradually (optional)."""
    return text

def welcome_message():
    return "👋 Welcome! InsightPilot helps you clean, analyze, and visualize data easily."

def introduction_message():
    return [
        "InsightPilot can help you clean data, perform AI-based analysis, and visualize insights automatically.",
        """
        Upload your dataset,  
        choose the mode,  
        and get results instantly — no coding needed!
        """
    ]
