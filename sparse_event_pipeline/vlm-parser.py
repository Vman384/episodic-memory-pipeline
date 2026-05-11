import os
import json
import time
import argparse
from google import genai
from typing import List, Any, Optional
import asyncio
import time

class AIEventParser:
     def __init__(self, model_name: str = "gemini-1.5-pro-latest", max_concurrent_task: int  = 3):
          """
          Initialise the class
          
          Args:
               model_name: The AI model in use
               max_concurrent_tasks: Max nmumber of sections to process at the same time.
                                     Default low to avoid errors (We're API key limited)
          """
          self.api_key = os.environ.get("GEMINI_API_KEY")
          if not self.api_key:
               raise ValueError("GEMINI_API_KEY not found, please provide valid api key")
          
          # semaphore to limit how many task can run concurrently
          self.semaphore = asyncio.Semaphore(max_concurrent_task)

          self.prompt = """

          You are a driver driving looking out for any rare or noteworthy events 
          in a forest, rural and tunnel road environments. We are looking for unusual 
          occurences, unexpected objects, events, sudden changes, or anything that stands 
          out from just regularly driving. DO NOT report normal driving, regular traffic, or 
          mundane things. We only want unusual, rare and interesting events 

          Output your findings in a strict JSON object with the follinwg structure. If nothing 
          happens, return an empty interesting events list
          
          {
               "video_name": "original video name here",
               "section_number": "chunk id here",
               "video_path": "path to chunk here",
               "interesting_events": [
                         {
                              "timestamp": "approximate timestamp in seconds within this chunk",
                              "event_description": "detailed description of what happened",
                              "why_interesting": "why this is considered a rare/sparse event"
                         }
                    ]
          }
          """


if __name__ == "__main__":
     parser = argparse.ArgumentParser("Get events from video sections")
     parser.add_argument("section_dir", help="Directory containing each sections for a unique video")
     parser.add_argument("--output_file", default="events.json", help="Path to save the output json")
     parser.add_argument("--model", default="gemini-1.5-pro-latest", help="Gemini model to use (default: gemini-1.5-pro)")

     args = parser.parse_args()