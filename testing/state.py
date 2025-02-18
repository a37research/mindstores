import minedojo
import numpy as np
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
from config import OPENAI_API_KEY
import cv2

class MinecraftStateObserver:
    def __init__(self):
        # Initialize MineDojo environment
        print("\nInitializing MineDojo environment...")
        self.env = minedojo.make(
            task_id="survival",
            image_size=(160, 256)
        )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def process_image(self, obs):
        """Process the observation image into a format suitable for GPT-4 Vision"""
        # Get RGB image and ensure it's in the correct format
        rgb_image = obs['rgb']
        
        # Convert to uint8 if not already
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # Ensure correct shape (height, width, channels)
        if rgb_image.shape[-1] != 3:  # If channels are not last
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        
        # Convert to BGR for cv2
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Convert back to RGB for PIL
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(rgb_image)

    def get_state_description(self, obs):
        """Get a description of the current state using GPT-4 Vision"""
        try:
            # Process the image
            image = self.process_image(obs)
            
            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare the message for GPT-4 Vision
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Describe what you see in this Minecraft scene in a single, detailed sentence. Include:
- Spatial descriptions (left, right, front, behind, above, below)
- Specific block types and their locations
- Distances when relevant (nearby, far)
- Any mobs or items and their positions
- Notable terrain features and their directions

Example: 'There's a oak tree to the left, stone blocks directly ahead about 10 blocks away, and a small cave entrance below with iron ore visible inside.'"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting scene description: {e}")
            return "Unable to analyze scene"

    def observe_environment(self, steps=10, delay=20):
        """Observe the environment for a number of steps"""
        obs = self.env.reset()
        print("\nStarting environment observation...")
        
        for step in range(steps):
            print(f"\nStep {step + 1}:")
            print(f"Position: {obs['location_stats']['pos']}")
            print(f"Pitch: {obs['location_stats']['pitch']}")
            print(f"Yaw: {obs['location_stats']['yaw']}")
            
            # Get and print scene description
            description = self.get_state_description(obs)
            print("\nScene Description:")
            print(description)
            
            # Take a no-op action with occasional look around
            action = self.env.action_space.no_op()
            if step % 3 == 0:  # Every third step, look around
                action[3] = np.random.randint(0, 24)  # Random camera rotation
            
            # Take step in environment
            obs, _, done, _ = self.env.step(action)
            
            if done:
                print("\nEpisode finished early")
                break
            
            # Wait a bit between observations
            for _ in range(delay):
                self.env.step(self.env.action_space.no_op())
        
        self.env.close()

if __name__ == "__main__":
    observer = MinecraftStateObserver()
    observer.observe_environment()
