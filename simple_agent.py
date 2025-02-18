import minedojo
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from itertools import chain
from config import OPENAI_API_KEY
import base64
from PIL import Image
from io import BytesIO
import cv2
import time
import sys
import json
import faiss
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ActionExperience:
    state: str
    task: str
    plan: List[str]
    outcome: str
    reward: float
    success: bool  # Track if action was successful
    embedding: np.ndarray

class ExperienceStore:
    def __init__(self, embedding_dim=1536):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.experiences: List[ActionExperience] = []
        
    def add_experience(self, experience: ActionExperience):
        if len(self.experiences) == 0:
            self.index = faiss.IndexFlatL2(len(experience.embedding))
        self.index.add(experience.embedding.reshape(1, -1))
        self.experiences.append(experience)
        print(f"\n[Experience Added] Total experiences: {len(self.experiences)}")
        print(f"State: {experience.state[:100]}...")
        print(f"Plan: {' '.join(experience.plan)}")
        print(f"Outcome: {experience.outcome}")
        print(f"Success: {'✓' if experience.success else '✗'}")
        print(f"Reward: {experience.reward}\n")
        
    def get_similar_experiences(self, query_embedding: np.ndarray, k: int = 3) -> List[ActionExperience]:
        if len(self.experiences) == 0:
            print("\n[Experience Search] No experiences in database yet")
            return []
        k = min(k, len(self.experiences))
        if k == 0:
            return []
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        valid_indices = [i for i in indices[0] if 0 <= i < len(self.experiences)]
        similar = [self.experiences[i] for i in valid_indices]
        
        if similar:
            print(f"\n[Experience Search] Found {len(similar)} similar experiences:")
            for i, exp in enumerate(similar, 1):
                print(f"\n{i}. Previous experience:")
                print(f"State: {exp.state[:100]}...")  # Show first 100 chars
                print(f"Plan: {' '.join(exp.plan)}")
                print(f"Outcome: {exp.outcome}")
                print(f"Reward: {exp.reward}")
        else:
            print("\n[Experience Search] No similar experiences found")
            
        return similar

class MinecraftAgent:
    def __init__(self):
        print("\nInitializing MineDojo environment...")
        self.env = minedojo.make(
            task_id="survival",
            image_size=(480, 768),
            seed=42,
        )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize environment and set camera
        self.env.reset()
        self.set_neutral_camera()
        
        # Initialize action state
        self.current_action = self.env.action_space.no_op()

        # Initialize experience store
        self.experience_store = ExperienceStore()
        self.last_state = None
        self.last_action_sequence = None
        self.current_goal = None  # Track current goal
        
        # Keep track of recent actions to prevent loops
        self.recent_actions = []  # Store last N actions
        self.max_recent_actions = 5  # How many recent actions to remember
        self.action_weights = {  # Weight multipliers for action types
            "forward": 1.0,
            "backward": 0.7,  # Slightly discourage backward movement
            "move_left": 1.0,
            "move_right": 1.0,
            "jump": 1.0,      # Normal movement
            "attack": 0.8,    # Resource gathering
            "use": 0.8,       # Interaction
            "craft": 0.4,     # Crafting
            "equip": 0.7,     # Tool use
            "place": 0.6,     # Building
        }
        
    def set_neutral_camera(self):
        action = self.env.action_space.no_op()
        self.env.step(action)
        
    def process_image(self, obs):
        """Process the observation image into a format suitable for GPT-4 Vision"""
        rgb_image = obs['rgb']
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        if rgb_image.shape[-1] != 3:
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    def get_state_description(self, obs):
        """Get a detailed description of the current state"""
        try:
            image = self.process_image(obs)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze the Minecraft scene and provide a structured description in the following format:

Terrain: [Describe ground level, slopes, drops, water/lava - be specific about distances]
Blocks: [List any notable blocks with their exact positions - e.g. "oak tree 3 blocks ahead", "stone to immediate left"]
Hazards: [List any dangerous areas with exact distances - e.g. "2-block drop 1 block ahead", "lava pool 5 blocks right"]
Resources: [List any valuable resources with exact positions and quantities - e.g. "3 iron ore blocks visible in cliff face ahead"]
Next Actions: [List 2-3 SPECIFIC actions to progress, with clear reasoning]
Example:
Terrain: Flat grass with 3-block drop 2 blocks ahead, stone cliff rises 4 blocks up to the right
Blocks: Oak tree directly ahead, 3 stone blocks exposed in cliff face to right
Hazards: 3-block drop starts 2 blocks ahead, no other dangers visible
Resources: Coal ore visible in cliff face (2 blocks), oak wood available from tree
Next Actions: 
1. Move right to reach stone cliff face
2. Mine exposed coal ore blocks for early resource gathering
3. Use wood from nearby tree to craft tools"""
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
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting scene description: {e}")
            return "Unable to analyze scene"

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI's embedding model"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(1536)
            
    def evaluate_outcome(self, initial_state: str, final_state: str, reward: float, done: bool) -> tuple[str, bool]:
        """Evaluate the outcome of an action sequence and determine success"""
        messages = [
            {
                "role": "system",
                "content": "Evaluate the outcome of a Minecraft action sequence. Consider changes in state, rewards, and completion."
            },
            {
                "role": "user",
                "content": f"""Initial state: {initial_state}
Final state: {final_state}
Reward: {reward}
Done: {done}

Evaluate:
1. What changed in the environment?
2. Was this action successful (true/false)?
3. Brief explanation of success/failure

Format response as: outcome|success|explanation"""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100
            )
            outcome, success_str, explanation = response.choices[0].message.content.strip().split("|")
            return outcome, success_str.lower() == "true"
        except Exception as e:
            print(f"Error evaluating outcome: {e}")
            return f"Reward: {reward}, Done: {done}", reward > 0

    def plan_action(self, state_description, inventory_info, goal_description=""):
        """Plan next action sequence based on state description and inventory"""
        print("\n[Planning Action]")
        print(f"Current State: {state_description[:100]}...")
        print(f"Goal: {goal_description}")
        print(f"Recent actions: {' -> '.join(self.recent_actions)}")
        
        # Update current goal
        self.current_goal = goal_description
        
        # Store current state for experience tracking
        self.last_state = state_description
        
        # Get similar experiences
        query_text = f"""Goal: {goal_description}
State: {state_description}
Task: Find actions that progress toward: {goal_description}"""
        
        query_embedding = self.get_embedding(query_text)
        similar_experiences = self.experience_store.get_similar_experiences(query_embedding)
        
        # Include experiences in prompt, highlighting successes and failures
        experience_text = ""
        if similar_experiences:
            experience_text = "\n\nRelevant past experiences (✓ success, ✗ failure):"
            for exp in similar_experiences:
                status = "✓" if exp.success else "✗"
                score = self.calculate_action_score(exp.plan, exp.reward, exp.success, state_description)
                experience_text += f"\n{status} State: {exp.state}\nActions: {' '.join(exp.plan)}\nOutcome: {exp.outcome}\nScore: {score:.2f}\n"
                if not exp.success:
                    experience_text += "WARNING: These actions failed, avoid them!\n"
                if self.is_action_repetitive(exp.plan):
                    experience_text += "WARNING: These actions would be repetitive!\n"
        else:
            experience_text = "\n\nNo past experiences yet. Start with safe exploration actions."
            
        # Add goal and recent actions context
        experience_text += f"\n\nCurrent Goal: {goal_description}"
        experience_text += f"\nRecent actions (avoid repeating): {' -> '.join(self.recent_actions)}"
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a Minecraft agent that plans action sequences based on the current goal: {goal_description}
IMPORTANT: 
1. ONE action per line, NO numbering or extra text
2. For movement commands, specify number of steps: forward 5, move_left 3, etc.
3. VERIFY targets are visible before actions like attack or use
4. Start with movement/looking commands to get to targets
5. Include specific numbers for all actions (e.g., 'attack 5' not just 'attack')

Available actions:
- forward [steps]     (e.g., forward 5)
- backward [steps]    (e.g., backward 3)
- move_left [steps]   (e.g., move_left 4)
- move_right [steps]  (e.g., move_right 2)
- look_horizontal [angle]
- attack [times]
- use
- equip [item]
- place [block]
- destroy

Example for getting wood:
look_horizontal -30
forward 5
attack 10

Example for mining:
forward 3
look_horizontal 20
attack 15""" + experience_text
            },
            {
                "role": "user",
                "content": f"""Current state: {state_description}
Current inventory: {inventory_info}
Goal: {goal_description}

Based on the current goal and state, plan 10 precise and continuous actions that will make progress toward the goal in a way that the vision of the agent is not drastically changed due to looking around.
Remember: ONE action per line, NO numbering, EXACT format as shown."""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300
            )
            action_sequence = response.choices[0].message.content.strip().split('\n')
            self.last_action_sequence = action_sequence
            return action_sequence
        except Exception as e:
            print(f"Error planning action: {e}")
            return ["forward"]

    def is_action_repetitive(self, action_sequence):
        """Check if action sequence would create a repetitive pattern"""
        # Convert action sequence to basic actions (strip arguments)
        basic_actions = [action.split()[0] for action in action_sequence]
        
        # Check if this would create a back-and-forth pattern
        if len(self.recent_actions) >= 2:
            # Detect direct reversal (e.g., forward->backward, left->right)
            opposites = {
                "forward": "backward",
                "backward": "forward",
                "move_left": "move_right",
                "move_right": "move_left",
                "look_horizontal 10": "look_horizontal -10",
                "look_horizontal -10": "look_horizontal 10"
            }
            
            last_action = self.recent_actions[-1].split()[0]
            for new_action in basic_actions:
                if new_action == opposites.get(last_action, ""):
                    print(f"Avoiding reversal: {last_action} -> {new_action}")
                    return True
        
        # Check for repeated single actions
        if len(self.recent_actions) >= 3:
            if all(a.split()[0] == basic_actions[0] for a in self.recent_actions[-3:]):
                print(f"Avoiding three consecutive {basic_actions[0]} actions")
                return True
        
        return False

    def calculate_action_score(self, action_sequence, reward, success, state_description):
        """Calculate a score for an action sequence based on rewards, weights, and goal alignment"""
        score = reward  # Start with the base reward
        
        # Apply action type weights
        for action in action_sequence:
            action_type = action.split()[0]
            weight = self.action_weights.get(action_type, 1.0)
            score *= weight
        
        # Penalize failed actions
        if not success:
            score *= 0.5
        
        # Penalize repetitive actions
        if self.is_action_repetitive(action_sequence):
            score *= 0.3
            
        # Check goal alignment using GPT
        if self.current_goal:
            alignment_score = self.evaluate_goal_alignment(state_description, action_sequence)
            score *= alignment_score
        
        return score
        
    def evaluate_goal_alignment(self, state_description, action_sequence):
        """Evaluate how well actions align with current goal"""
        messages = [
            {
                "role": "system",
                "content": """You are a scoring system that evaluates action alignment with goals.
IMPORTANT: You must ONLY return a single number between 0.0 and 2.0, where:
0.0 = completely diverging from goal
1.0 = neutral/indirect progress
2.0 = perfect alignment with goal

Example responses:
"1.5"     (good progress toward goal)
"0.8"     (slightly misaligned)
"2.0"     (perfect alignment)

DO NOT provide explanations or text. ONLY return a single number."""
            },
            {
                "role": "user",
                "content": f"""Score this action sequence based on goal alignment:

Goal: {self.current_goal}
Current State: {state_description}
Actions: {' -> '.join(action_sequence)}

Scoring criteria:
- How directly do actions progress toward goal? (0.0-2.0)
- Are actions necessary for goal? (0.0-2.0)
- Do actions maintain progress toward goal? (0.0-2.0)

Return ONLY a single number between 0.0 and 2.0. No text."""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=10  # Very short response
            )
            response_text = response.choices[0].message.content.strip()
            
            # Extract the first number found in the response
            import re
            number_match = re.search(r'(\d*\.?\d+)', response_text)
            if number_match:
                score = float(number_match.group(1))
                return max(0.1, min(2.0, score))  # Clamp between 0.1 and 2.0
            
            print(f"Could not extract number from response: {response_text}")
            return 1.0  # Neutral score if no number found
            
        except Exception as e:
            print(f"Error evaluating goal alignment: {e}")
            return 1.0  # Neutral score on error

    def update_recent_actions(self, action_sequence):
        """Update the list of recent actions"""
        self.recent_actions.extend(action_sequence)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[-self.max_recent_actions:]
        print(f"\nRecent actions: {' -> '.join(self.recent_actions)}")

    def store_experience(self, final_state: str, reward: float, done: bool):
        """Store the experience from the last action sequence"""
        if self.last_state and self.last_action_sequence:
            print("\n[Storing Experience]")
            print(f"Initial State: {self.last_state[:100]}...")
            print(f"Action Sequence: {' '.join(self.last_action_sequence)}")
            
            outcome, success = self.evaluate_outcome(self.last_state, final_state, reward, done)
            print(f"Evaluated Outcome: {outcome}")
            print(f"Success: {'✓' if success else '✗'}")
            
            # Create embedding combining state, actions, and outcome
            context = f"""State: {self.last_state}
Actions: {' '.join(self.last_action_sequence)}
Outcome: {outcome}
Success: {success}
Reward: {reward}"""
            
            experience = ActionExperience(
                state=self.last_state,
                task="explore_and_gather",  # Default task
                plan=self.last_action_sequence,
                outcome=outcome,
                reward=reward,
                success=success,
                embedding=self.get_embedding(context)
            )
            
            self.experience_store.add_experience(experience)
            
            # Reset tracking
            self.last_state = None
            self.last_action_sequence = None

    def execute_action_sequence(self, action_sequence):
        """Execute a sequence of actions"""
        # Update recent actions before execution
        self.update_recent_actions(action_sequence)
        
        obs = None
        reward = 0
        done = False
        info = {}
        
        for action_str in action_sequence:
            # Parse action and args
            parts = action_str.strip().split()
            action_name = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Get action generator
            action_gen = None
            if action_name == "move_forward" or action_name == "forward":
                steps = int(args[0]) if args else 1
                action_gen = self.forward(steps)
            elif action_name == "backward":
                steps = int(args[0]) if args else 1
                action_gen = self.backward(steps)
            elif action_name == "move_left":
                steps = int(args[0]) if args else 1
                action_gen = self.move_left(steps)
            elif action_name == "move_right":
                steps = int(args[0]) if args else 1
                action_gen = self.move_right(steps)
            elif action_name == "jump":
                action_gen = self.jump()
            elif action_name == "sneak":
                action_gen = self.sneak()
            elif action_name == "sprint":
                action_gen = self.sprint()
            elif action_name == "use":
                action_gen = self.use()
            elif action_name == "drop":
                action_gen = self.drop()
            elif action_name == "attack":
                # Verify target before attacking
                obs = self.env.step(self.env.action_space.no_op())[0]
                if not self.verify_target_visible(obs, "tree" if "wood" in " ".join(args) else "block"):
                    print("Target not visible, skipping attack")
                    continue
                times = int(args[0]) if args else 5
                action_gen = self.attack(times)
            elif action_name == "craft":
                action_gen = self.craft()
            elif action_name == "equip" and args:
                item_name = " ".join(args)  # Handle multi-word items
                action_gen = self.equip(item_name)
            elif action_name == "place" and args:
                block_name = " ".join(args)  # Handle multi-word blocks
                action_gen = self.place(block_name)
            elif action_name == "destroy":
                action_gen = self.destroy()
            elif action_name == "look_vertical" and args:
                try:
                    angle = float(args[0])
                    action_gen = self.look_vertical(angle)
                except ValueError:
                    print(f"Invalid angle: {args[0]}")
                    continue
            elif action_name == "look_horizontal" and args:
                try:
                    angle = float(args[0])
                    action_gen = self.look_horizontal(angle)
                except ValueError:
                    print(f"Invalid angle: {args[0]}")
                    continue
            elif action_name == "no_op":
                action_gen = self.no_op()
            else:
                print(f"Unknown action: {action_name}")
                continue
            
            # Execute action for a few steps
            if action_gen:
                for action in action_gen:
                    print(f"Executing action: {action}")
                    # Execute action multiple times for smooth movement
                    for _ in range(5):
                        obs, reward, done, info = self.env.step(action)
                        if done:
                            print("Episode finished during action")
                            self.store_experience(self.get_state_description(obs), reward, done)
                            return obs, reward, done, info
                    
                    if done:
                        print("Episode finished during reset")
                        self.store_experience(self.get_state_description(obs), reward, done)
                        return obs, reward, done, info
        
        # Store experience after completing sequence
        if obs is not None:
            self.store_experience(self.get_state_description(obs), reward, done)
        return obs, reward, done, info

    def forward(self, steps=1):
        """Move forward specified number of steps"""
        steps = int(steps)
        for _ in range(steps):
            act = self.env.action_space.no_op()
            act[0] = 1
            yield act
            # Check if we hit something
            obs, _, _, info = self.env.step(self.env.action_space.no_op())
            if self.is_blocked(obs):
                print("Movement blocked, stopping forward motion")
                break

    def backward(self, steps=1):
        """Move backward specified number of steps"""
        steps = int(steps)
        for _ in range(steps):
            act = self.env.action_space.no_op()
            act[0] = 2
            yield act
            obs, _, _, info = self.env.step(self.env.action_space.no_op())
            if self.is_blocked(obs):
                print("Movement blocked, stopping backward motion")
                break

    def move_left(self, steps=1):
        """Move left specified number of steps"""
        steps = int(steps)
        for _ in range(steps):
            act = self.env.action_space.no_op()
            act[1] = 1
            yield act
            obs, _, _, info = self.env.step(self.env.action_space.no_op())
            if self.is_blocked(obs):
                print("Movement blocked, stopping left motion")
                break

    def move_right(self, steps=1):
        """Move right specified number of steps"""
        steps = int(steps)
        for _ in range(steps):
            act = self.env.action_space.no_op()
            act[1] = 2
            yield act
            obs, _, _, info = self.env.step(self.env.action_space.no_op())
            if self.is_blocked(obs):
                print("Movement blocked, stopping right motion")
                break

    def is_blocked(self, obs):
        """Check if movement is blocked"""
        # You might need to adjust this based on MineDojo's observation format
        try:
            # Check if there's a block directly in front
            ray_cast = obs.get('ray_cast', {})
            if ray_cast.get('distance', 100) < 0.5:  # If something is very close
                return True
            return False
        except Exception as e:
            print(f"Error checking if blocked: {e}")
            return False

    def verify_target_visible(self, obs, target_type):
        """Verify if target (like a tree) is visible in current observation"""
        try:
            # Use vision model to check if target is visible
            description = self.get_state_description(obs)
            messages = [
                {
                    "role": "system",
                    "content": "You are a vision system that verifies if specific objects are visible. Return ONLY 'yes' or 'no'."
                },
                {
                    "role": "user",
                    "content": f"Is there a {target_type} visible in this scene? Scene description: {description}\n\nAnswer only 'yes' or 'no'."
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=10
            )
            return response.choices[0].message.content.strip().lower() == "yes"
        except Exception as e:
            print(f"Error verifying target visibility: {e}")
            return False

    def run_agent(self, max_steps=1000, goal_description=""):
        """Main agent loop"""
        print("\nStarting agent loop...")
        
        # Initialize with centered camera
        init_action = self.env.action_space.no_op()
        obs, _, _, info = self.env.step(init_action)
        
        step = 0
        
        while step < max_steps:
            # Ensure agent is completely still during planning
            # Get current state
            state_desc = self.get_state_description(obs)
            print(f"\nStep {step + 1}:")
            print(f"State: {state_desc}")
            
            # Get inventory info from last observation's info
            inventory = info['inventory']
            
            # Plan next action sequence using current state
            action_sequence = self.plan_action(state_desc, inventory, goal_description)
            print(f"Planned action sequence:")
            for action in action_sequence:
                print(f"- {action}")
            
            # Execute action sequence
            obs, reward, done, info = self.execute_action_sequence(action_sequence)
            self.env.action_space.no_op()
            step += len(action_sequence)
            
            if done:
                print("\nEpisode finished")
                break
        
        self.env.close()

    def attack(self, times = 2):
        for i in range(times):
            act = self.env.action_space.no_op()
            act[5] = 3
            yield act
        yield self.env.action_space.no_op()

    def use(self):
        act = self.env.action_space.no_op()
        act[5] = 1
        yield act
        yield self.env.action_space.no_op()

    def drop(self):
        """Drop item"""
        act = self.env.action_space.no_op()
        act[5] = 2
        yield act

    def craft(self):
        """Craft item"""
        act = self.env.action_space.no_op()
        act[5] = 4
        yield act
    def place(self, goal):
        slot = self.index_slot(goal)
        if slot == -1:
            return False
        act = self.env.action_space.no_op()
        act[5] = 6
        act[7] = slot
        yield act

    def destroy(self):
        """Destroy block"""
        act = self.env.action_space.no_op()
        act[5] = 7
        yield act

    def sneak(self, times=1):
        """Move forward carefully"""
        for _ in range(times):
            act = self.env.action_space.no_op()
            act[0] = 1  # Slower forward movement
            yield act

    def smelt(self, item):
        """Smelt items in furnace"""
        # Simplified smelting - just place and use furnace
        for act in chain(
            self.place("furnace"),
            self.use()
        ):
            yield act

    def no_op(self):
        act = self.env.action_space.no_op()
        yield act

    def jump(self):
        act = self.env.action_space.no_op()
        act[2] = 1
        yield act
        yield self.env.action_space.no_op()

    def jump_forward(self):
        """Jump while moving forward"""
        act = self.env.action_space.no_op()
        act[0] = 1  # Forward
        act[2] = 1  # Jump
        yield act
        yield self.env.action_space.no_op()

    def equip(self, goal):
        obs, reward, done, info = self.acquire_info()
        for item in info['inventory']:
            if item['name'] == goal and item['index'] > 0:
                act = self.env.action_space.no_op()
                act[5] = 5
                act[7] = item['index']
                yield act
                return 

    def look_vertical(self, angle_degrees):
        """Look up or down by specified angle.
        angle_degrees: float between -180 and 180
        -180 is straight down, 0 is center, 180 is straight up"""
        # Convert angle to 0-24 range
        # -180 maps to 0, 0 maps to 12, 180 maps to 24
        normalized = (angle_degrees + 180) / 360 * 24
        camera_pos = int(round(normalized))
        # Clamp to valid range
        camera_pos = max(0, min(24, camera_pos))
        
        act = self.env.action_space.no_op()
        act[3] = camera_pos  # Pitch
        yield act

    def look_horizontal(self, angle_degrees):
        """Look left or right by specified angle.
        angle_degrees: float between -180 and 180
        -180 is full left, 0 is center, 180 is full right"""
        # Convert angle to 0-24 range
        # -180 maps to 0, 0 maps to 12, 180 maps to 24
        normalized = (angle_degrees + 180) / 360 * 24
        camera_pos = int(round(normalized))
        # Clamp to valid range
        camera_pos = max(0, min(24, camera_pos))
        
        act = self.env.action_space.no_op()
        act[4] = camera_pos  # Yaw
        yield act

    def acquire_info(self):
        return self.env.step(self.env.action_space.no_op())
    def index_slot(self, goal):
        #! accquire info 
        obs, reward, done, info = self.acquire_info()
        slot = -1
        for item in info['inventory']:
            if goal == item['name']:
                slot = item['index']
                break
        return slot
    def equip(self, goal):
        obs, reward, done, info = self.acquire_info()
        for item in info['inventory']:
            if item['name'] == goal and item['index'] > 0:
                act = self.env.action_space.no_op()
                act[5] = 5
                act[7] = item['index']
                yield act
                return 

if __name__ == "__main__":
    # Get goal description from command line args if provided
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    
    agent = MinecraftAgent()
    agent.run_agent(goal_description=goal)
