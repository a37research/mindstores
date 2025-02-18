import minedojo
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from itertools import chain
import base64
import cv2
import faiss
import json
import sys
from dataclasses import dataclass
from minedojo.sim import InventoryItem
from typing import List, Dict, Optional
from PIL import Image
from io import BytesIO
from config import OPENAI_API_KEY
import time
import logging
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

####################################
# EXPERIENCE STORE + DATA STRUCTURE
####################################
@dataclass
class ActionExperience:
    state: Dict
    task: str
    plan: List[str]
    outcome: str
    success: bool
    reward: float
    embedding: np.ndarray

class ExperienceStore:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vision_encoder = CLIPVisionEncoder()
        self.index = faiss.IndexFlatL2(768 + 512)
        self.experiences: List[ActionExperience] = []
        self.logger = logging.getLogger('MinecraftAgent.ExperienceStore')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_proj = torch.nn.Linear(768, 768)
        self.vis_proj = torch.nn.Linear(512, 512)

    def encode_experience(self, experience: ActionExperience):
        text_embed = self.text_encoder.encode(experience.state['text'])
        text_embed = torch.FloatTensor(text_embed)
        visual_embed = experience.state['visual']
        visual_embed = torch.FloatTensor(visual_embed)
        
        # Apply projection and attention
        text_att = torch.softmax(self.text_proj(text_embed), dim=-1)
        vis_att = torch.softmax(self.vis_proj(visual_embed), dim=-1)
        combined = torch.cat([text_att * text_embed, vis_att * visual_embed]).numpy()
        return combined / np.linalg.norm(combined)

    def add_experience(self, experience: ActionExperience):
        if experience.embedding is None:
            experience.embedding = self.encode_experience(experience)
        self.experiences.append(experience)
        self.index.add(experience.embedding.reshape(1, -1))
        self.logger.info(f"New Experience Added: {experience.task}")
        self.check_database_health()

    def find_similar(self, query_state: dict, k=5):
        text_embed = self.text_encoder.encode(query_state['text'])
        visual_embed = query_state['visual']
        query_text_embed = torch.FloatTensor(text_embed)
        query_vis_embed = torch.FloatTensor(visual_embed)
        
        # Apply attention mechanisms
        text_att = torch.softmax(self.text_proj(query_text_embed), dim=-1)
        vis_att = torch.softmax(self.vis_proj(query_vis_embed), dim=-1)
        fused_embed = torch.cat([text_att * query_text_embed, vis_att * query_vis_embed])
        _, indices = self.index.search(fused_embed.reshape(1,-1).numpy(), k)
        return [self.experiences[i] for i in indices[0]]

    def check_database_health(self):
        db_size = len(self.experiences)
        self.logger.info(f"Experience DB size: {db_size}")

####################################
# VISION PROCESSING ENHANCEMENTS
####################################
class CLIPVisionEncoder:
    def __init__(self):
        self.model, _ = clip.load('ViT-B/32', device='cpu')
        self.preprocess = Compose([
            Resize(224, interpolation=3),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.labels = [
            "tree", "stone", "water", "animal", "building", 
            "tool", "enemy", "path", "cave", "ore"
        ]

    def encode_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        with torch.no_grad():
            image = self.preprocess(image).unsqueeze(0)
            return self.model.encode_image(image).cpu().numpy().squeeze()

    def describe_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = self.preprocess(image).unsqueeze(0)
        text_features = self.model.encode_text(clip.tokenize(self.labels).to('cpu'))
        image_features = self.model.encode_image(image)
        
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        top_indices = similarity[0].topk(3).indices
        return [self.labels[i] for i in top_indices]

####################################
# MINECRAFT AGENT IMPROVEMENTS
####################################
class MinecraftAgent:
    def __init__(self, goal_input: str = "Explore and gather resources"):
        # Set up logging
        try:
            base_dir = os.path.dirname(__file__)
        except NameError:
            base_dir = "."
        log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'minecraft_agent_{timestamp}.log')
        
        # Configure logging
        self.logger = logging.getLogger('MinecraftAgent')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Initializing MineDojo environment...")
        print("\nInitializing MineDojo environment...")

        # 1) Create the MineDojo environment
        self.env = minedojo.make(
            task_id="survival",
            image_size=(480, 768),
            seed=40,
            initial_inventory = [
                InventoryItem(slot=0, name="wooden_axe", variant=None, quantity=1),
            ]
        )
        
        # 2) Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 3) Reset environment
        obs = self.env.reset()
        
        # 4) Set neutral camera position
        self.set_neutral_camera()
        
        # Experience store
        self.experience_store = ExperienceStore()
        
        # Track the last state, action sequence, and raw GPT plan
        self.last_state = None
        self.last_action_sequence = None
        self.last_gpt_plan = None
        
        # We store the high-level user goal
        self.user_goal = goal_input
        
        # Keep track of recent actions to prevent loops
        self.recent_actions = []
        self.max_recent_actions = 5
        
        # Action weights
        self.action_weights = {
            "forward": 1.0,
            "backward": 0.7,
            "move_left": 1.0,
            "move_right": 1.0,
            "jump": 1.0,
            "attack": 0.8,
            "use": 0.8,
            "craft": 0.4,
            "equip": 0.7,
            "place": 0.6,
            "destroy": 0.7,
        }
        
        # Metrics logger
        self.metrics_logger = MetricsLogger()
        
        # Vision model initialization
        self.vision_encoder = CLIPVisionEncoder()
        
        self.logger.info("Agent initialization complete")

    def process_image(self, obs):
        rgb = obs['rgb']
        rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (224, 224))
        pil_img = Image.fromarray(rgb)
        
        # Get both embedding and description
        visual_embed = self.vision_encoder.encode_image(pil_img)
        visual_description = self.vision_encoder.describe_image(pil_img)
        return visual_embed, visual_description

    def create_structured_state(self, obs, info, current_sub_task=""):
        visual_embed, visual_desc = self.process_image(obs)
        text_state = {
            'inventory': info.get('inventory', []),
            'health': info.get('health', 20),
            'position': info.get('position', (0,0,0)),
            'visual_description': visual_desc,
            'current_task': current_sub_task
        }
        return {
            'text': json.dumps(text_state),
            'visual': visual_embed,
            'timestamp': datetime.now().isoformat()
        }

    def describe_environment(self, state_json_str: str) -> str:
        state = json.loads(state_json_str)
        visual_desc = ", ".join(state.get('visual_description', []))
        prompt = f"""Visual scene contains: {visual_desc}. Describe the environment focusing on:
1. Immediate surroundings and resources
2. Potential threats or opportunities
3. Navigation possibilities
4. Relevant objects for current goals"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content

    def explain_situation(self, state_json_str: str, description: str) -> str:
        prompt = f"""You are an expert Minecraft strategist. Given the current state and environment description:
1. Analyze available resources and their potential uses
2. Identify immediate opportunities or threats
3. Consider crafting possibilities based on inventory
4. Evaluate progress towards goals

Environment description:
{description}

Current state:
{state_json_str}

Provide strategic insights about the current situation."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content

    def plan_strategy(self, state_json_str: str, description: str, explanation: str, goal: str) -> str:
        prompt = f"""You are an expert Minecraft planner. Create a strategic plan considering:
1. The current goal: {goal}
2. Available resources and tools
3. Environmental conditions
4. Potential obstacles or requirements
5. Do not assume that intermediately in this plan that important tasks can be achieved by the end of it without running another loop of the agent. For example, do not say move forward, attack, and then craft knowing that this would not be gauranteed to work.
6. make sure to give some quanitity of the possible distance and the actions needed for this.

Environment description:
{description}

Situation analysis:
{explanation}

Current state:
{state_json_str}

Create a specific, actionable plan that moves towards the goal."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content

    def select_actions(self, state_json_str: str, plan: str) -> List[str]:
        prompt = f"""You are an expert Minecraft action selector. Convert the plan into specific actions:
1. Use only valid Minecraft actions (move_forward, move_backward, jump, craft, etc.)
2. Consider the current state and available resources as proposed beforehand
3. Break down complex tasks into simple action sequences
4. Ensure actions are feasible given the agent's capabilities
5. Make sure that each action is VERY incremental and builds on top of each other, do not make large jumps in tasks like move forward, jump, attack, and then craft not knowing that we have something to craft.
- Available actions:
- forward [N]: Move forward N steps (default 1)
- backward [N]: Move backward N steps (default 1)
- move_left
- move_right
- jump
- sneak
- sprint
- attack [N]
- use
- drop
- craft
- equip [item]
- place [block]
- destroy
- look_horizontal +/-X
- no_op
Example:
Let's say the subtask is that a tree is within the vicinity of 12 steps forward and 4 steps to the right, a valid and good plan would be to:
- move_forward 8
- look_horizontal +30
- move_right 2
- move_forward 4
- attack 4
A bad plan would be:
- move_forward 1
- attack 1
- craft wooden_axe 1
- equip wooden_axe
- attack 1

Strategic plan:
{plan}

Current state:
{state_json_str}

Return ONLY a list of actions, one per line, that can be directly executed."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return [action.strip() for action in response.choices[0].message.content.split('\n') if action.strip()]

    def get_next_immediate_task(self, state_str: str) -> str:
        """
        Enhanced task selection using the specialized LLMs pipeline.
        """
        # Get environment description
        description = self.describe_environment(state_str)
        
        # Analyze the situation
        explanation = self.explain_situation(state_str, description)
        
        # Create a strategic plan
        plan = self.plan_strategy(state_str, description, explanation, self.user_goal)
        
        # Select specific actions
        actions = self.select_actions(state_str, plan)
        
        self.logger.info(f"""
Environment Description:
{description}

Situation Analysis:
{explanation}

Strategic Plan:
{plan}

Selected Actions:
{json.dumps(actions, indent=2)}
""")
        
        return actions[0] if actions else "look"  # Default to look if no actions selected

    ########################################
    # 3) plan_action
    ########################################
    def plan_action(self, state_json_str, inventory_info, sub_goal: str):
        self.last_state = state_json_str
        
        # Experience-based context
        query_text = f"Goal: {sub_goal}\nState: {state_json_str}"
        query_embedding = self.get_embedding(query_text)
        
        # Time the retrieval
        start_retrieval = time.time()
        similar_exps = self.experience_store.find_similar(state_json_str, sub_goal)
        retrieval_time_ms = (time.time() - start_retrieval) * 1000
        
        # Log retrieval
        self.metrics_logger.log_experience_retrieval(
            retrieved_count=len(similar_exps),
            db_size=len(self.experience_store.experiences),
            retrieval_time_ms=retrieval_time_ms
        )
        
        experience_text = "No relevant experiences.\n"
        if len(similar_exps) > 0:
            experience_text = "Relevant Past Experiences:\n"
            for exp in similar_exps:
                status = "✓" if exp.success else "✗"
                experience_text += f"{status} Plan: {exp.plan} | Outcome: {exp.outcome}\n"

        system_prompt = f"""You are a specialized Minecraft action-planning agent. 
You have a sub-goal: {sub_goal}

Produce EXACTLY 10 lines of discrete Minecraft actions, one per line, 
with no extra text or numbering. Available actions:
- forward [N]: Move forward N steps (default 1)
- backward [N]: Move backward N steps (default 1)
- move_left
- move_right
- jump
- sneak
- sprint
- attack [N]
- use
- drop
- craft
- equip [item]
- place [block]
- destroy
- look_horizontal +/-X
- no_op

Rules:
1) Avoid repeating the same action 3+ times in a row.
2) 'destroy' only if block is in front, 'use' only if there's an interactable in front.
3) 'forward N' means move forward N times, likewise 'backward N'.
4) 'attack N' means attack N times.
5) Return EXACTLY 10 lines.

{experience_text}
"""

        user_prompt = f"""
[ENV STATE]:
{state_json_str}

[INVENTORY]:
{inventory_info}

[RECENT ACTIONS]:
{self.recent_actions}

Sub-goal: {sub_goal}
Please produce 10 lines of final actions.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300
            )
            # Store the raw GPT plan lines in self.last_gpt_plan
            raw_plan_lines = response.choices[0].message.content.strip().split('\n')
            self.last_gpt_plan = raw_plan_lines[:]  # Copy the list

            final_actions = self.post_process_actions(raw_plan_lines, state_json_str)
            self.last_action_sequence = final_actions

            print(f"[METRICS] Plan refinement iterations: 1")
            print(f"[METRICS] Final plan length: {len(final_actions)}")

            return final_actions
        except Exception as e:
            print(f"[plan_action] GPT Error: {e}")
            return ["forward"]

    ########################################
    # 4) POST-PROCESS
    ########################################
    def post_process_actions(self, actions, state_json_str):
        """Simple checks to skip impossible actions, or fill to 10 lines."""
        try:
            state_dict = json.loads(state_json_str)
        except:
            return actions
        
        # Parse the inventory_list from the structured state
        inventory_list = state_dict.get("inventory_list", [])
        inv_dict = {}
        for it in inventory_list:
            inv_dict[it["name"]] = it["quantity"]
        
        cleaned = []
        for act in actions:
            parts = act.strip().split()
            if not parts:
                continue
            name = parts[0].lower()
            
            # skip place if we don't have the block in inventory
            if name == "place" and len(parts) > 1:
                block_name = " ".join(parts[1:])
                if block_name not in inv_dict or inv_dict[block_name] <= 0:
                    print(f"Post-process: removing 'place {block_name}' - not in inventory.")
                    continue
            
            cleaned.append(act)

        # ensure exactly 10 lines
        cleaned = cleaned[:10]
        while len(cleaned) < 10:
            cleaned.append("no_op")
        return cleaned

    ########################################
    # 5) EMBEDDINGS & OUTCOME
    ########################################
    def get_embedding(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(1536)

    def evaluate_outcome(
        self,
        initial_state: str,
        final_state: str,  
        reward: float,
        done: bool,
        gpt_plan: list[str] = None,    
        executed_actions: list[str] = None 
    ) -> tuple[str, bool]:
        """
        Evaluate the outcome of a Minecraft action sequence in brief, 
        now including the GPT plan and executed actions in the prompt.
        """
        if gpt_plan is None:
            gpt_plan = []
        if executed_actions is None:
            executed_actions = []

        system_msg = "Evaluate the outcome of a Minecraft action sequence in brief."
        user_msg = f"""
Initial state (JSON): {initial_state}
Final state (JSON): {final_state}
Reward: {reward}
Done: {done}

GPT Plan: {gpt_plan}
Executed Actions: {executed_actions}

Format response as: outcome|success|explanation
""" 
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=100
            )
            text = response.choices[0].message.content.strip()
            outcome, success_str, explanation = text.split("|")
            return outcome, (success_str.lower() == "true")
        except Exception as e:
            print(f"Error evaluating outcome: {e}")
            return f"Reward: {reward}, Done: {done}", (reward > 0)

    ########################################
    # 6) RUNTIME LOOP
    ########################################
    def run_agent(self, max_steps=1000):
        print("\nStarting agent loop...")
        obs, _, _, info = self.env.step(self.env.action_space.no_op())
        step = 0
        
        while step < max_steps:
            # Build structured state JSON
            state_json_str = self.get_state_description(obs, info)  # No sub-task known yet

            # 1) Produce a sub-task
            sub_task = self.get_next_immediate_task(state_json_str)
            print(f"[LLM Sub-Task]\nTask: {sub_task}")

            # Start subtask timer
            self.metrics_logger.start_subtask()

            # 2) Plan an action sequence (pass in the sub_task for logging)
            # Rebuild the structured state so that 'current_task' is set
            state_json_str = self.get_state_description(obs, info, sub_task)
            inventory = info.get("inventory", [])
            actions = self.plan_action(state_json_str, inventory, sub_task)
            print(f"\n[GPT Planned Actions for '{sub_task}']:\n" + "\n".join(actions))
            
            # 3) Execute
            obs, reward, done, info = self.execute_action_sequence(actions, obs, info)
            step += len(actions)

            if done:
                print("\nEpisode finished.")
                self.metrics_logger.end_subtask(success=False)
                break
            
            self.metrics_logger.end_subtask(success=False)

        self.env.close()
        self.metrics_logger.print_summary()

    def update_recent_actions(self, action_sequence):
        self.recent_actions.extend(action_sequence)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[-self.max_recent_actions:]
        print(f"Recent actions: {self.recent_actions}")

    def store_experience(self, final_state: str, reward: float, done: bool):
        if self.last_state and self.last_action_sequence:
            raw_gpt_plan = self.last_gpt_plan
            
            self.logger.info(f"""
Evaluating Experience:
Initial State: {self.last_state}
Final State: {final_state}
Reward: {reward}
Done: {done}
""")

            outcome_text, success = self.evaluate_outcome(
                self.last_state,
                final_state,
                reward,
                done,
                raw_gpt_plan,
                self.last_action_sequence
            )

            context = f"""
Initial State: {self.last_state}
Actions Planned: {' '.join(raw_gpt_plan) if raw_gpt_plan else 'None'}
Actions Executed: {' '.join(self.last_action_sequence)}
Final State: {final_state}
Outcome: {outcome_text}
"""
            embedding = self.experience_store.encoder.encode(context, normalize_embeddings=True)
            
            self.logger.info(f"""
Generated Embedding:
Norm: {np.linalg.norm(embedding):.2f}
Shape: {embedding.shape}
""")

            experience = ActionExperience(
                state=self.last_state,
                task=self.user_goal,  # Use the high-level user goal as the task
                plan=raw_gpt_plan,
                outcome=outcome_text,
                success=success,
                reward=reward,
                embedding=embedding
            )
            
            self.experience_store.add_experience(experience)
            
            # Update task completion status
            try:
                state_dict = json.loads(final_state)
                current_task = state_dict.get("current_task", "")
                if success and current_task:
                    self.add_completed_task(current_task)
                elif not success and current_task:
                    self.add_failed_task(current_task)
            except json.JSONDecodeError:
                self.logger.warning("Could not parse final state to update task status")
            
            if success:
                self.logger.info("Subtask completed successfully")
                self.metrics_logger.end_subtask(success=True)
            else:
                self.logger.info("Subtask completed without success")
                self.metrics_logger.end_subtask(success=False)

            # Reset references so we don't reuse them inadvertently
            self.last_state = None
            self.last_action_sequence = None
            self.last_gpt_plan = None  # Clear the raw plan

    ###################################
    # [NEW LOGIC ADDED] - Check block in front
    ###################################
    def is_block_in_front(self, obs, info) -> bool:
        """
        Minimal placeholder function to check if there's a non-air block in front
        based on the voxels' 'cos_look_vec_angle'. If angle is close to 1 
        and block is not 'minecraft:air', we consider it "in front".
        """
        block_names = obs["voxels"]["block_name"]
        cos_angles = obs["voxels"]["cos_look_vec_angle"]
        
        # Convert to numpy arrays if they aren't already
        if isinstance(block_names, list):
            block_names = np.array(block_names)
        if isinstance(cos_angles, list):
            cos_angles = np.array(cos_angles)
            
        # Find blocks that are in front (cos angle > 0.99) and not air
        non_air_mask = block_names != "minecraft:air"
        in_front_mask = cos_angles > 0.99
        
        return np.any(non_air_mask & in_front_mask)

    ###################################
    # EXECUTE ACTIONS
    ###################################
    def execute_action_sequence(self, action_sequence, obs, info):
        self.update_recent_actions(action_sequence)
        
        total_reward = 0
        done = False
        
        for action_str in action_sequence:
            parts = action_str.strip().split()
            if not parts:
                continue
            action_name = parts[0].lower()
            args = parts[1:]
            
            action_gen = None

            if action_name == "forward" or action_name == "move_forward":
                steps = 1
                if args:
                    try:
                        steps = int(args[0])
                    except:
                        pass
                action_gen = self.forward(obs, steps)

            elif action_name == "backward" or action_name == "move_backward":
                steps = 1
                if args:
                    try:
                        steps = int(args[0])
                    except:
                        pass
                action_gen = self.backward(obs, steps)

            elif action_name == "move_left":
                action_gen = self.move_left()

            elif action_name == "move_right":
                action_gen = self.move_right()

            elif action_name == "jump":
                action_gen = self.jump()

            elif action_name == "sneak":
                times = 1
                if args:
                    try:
                        times = int(args[0])
                    except:
                        pass
                action_gen = self.sneak(times)

            elif action_name == "sprint":
                action_gen = self.sprint()

            elif action_name == "attack":
                times = 2
                if args:
                    try:
                        times = int(args[0])
                    except:
                        pass
                action_gen = self.attack(times)

            elif action_name == "use":
                action_gen = self.use()

            elif action_name == "drop":
                action_gen = self.drop()

            elif action_name == "craft":
                action_gen = self.craft()

            elif action_name == "equip":
                item_name = " ".join(args)
                action_gen = self.equip(item_name)

            elif action_name == "place":
                block_name = " ".join(args)
                action_gen = self.place(block_name)

            elif action_name == "destroy":
                if self.is_block_in_front(obs, info):
                    action_gen = self.destroy()
                else:
                    self.logger.debug("No block in front, skipping destroy")
                    continue

            elif action_name == "look_horizontal":
                if args:
                    try:
                        deg = float(args[0])
                        action_gen = self.look_horizontal(deg)
                    except:
                        pass
            elif action_name == "no_op":
                action_gen = self.no_op()

            else:
                self.logger.warning(f"Unknown action: {action_str}")
                continue
            
            if action_gen:
                for act in action_gen:
                    obs, step_reward, done, info = self.env.step(act)
                    total_reward += step_reward
                    if done:
                        print("\nEpisode finished mid-action.")
                        self.store_experience(self.get_state_description(obs, info), total_reward, done)
                        return obs, total_reward, done, info

                if done:
                    self.store_experience(self.get_state_description(obs, info), total_reward, done)
                    return obs, total_reward, done, info
        
        # If not done by the end of all actions, store experience
        self.store_experience(self.get_state_description(obs, info), total_reward, done)
        return obs, total_reward, done, info

    ###################################
    # BASIC ACTION DEFINITIONS
    ###################################
    def forward(self, obs, steps=1):
        self.logger.info(f"Moving forward {steps} steps")
        for _ in range(steps):
            if not self.is_blocked(obs):
                action = self.create_action()
                action[0] = 1  # Move forward
                yield action
            else:
                self.logger.debug("Movement blocked, stopping forward")
                break

    def backward(self, obs, steps=1):
        self.logger.info(f"Moving backward {steps} steps")
        for _ in range(steps):
            if not self.is_blocked(obs):
                action = self.create_action()
                action[0] = 2  # Move backward
                yield action
            else:
                self.logger.debug("Movement blocked, stopping backward")
                break

    def move_left(self):
        action = self.create_action()
        action[1] = 1  # Move left (1 in the left/right channel)
        yield action

    def move_right(self):
        action = self.create_action()
        action[1] = 2  # Move right (2 in the left/right channel)
        yield action

    def jump(self):
        action = self.create_action()
        action[2] = 1  # Jump
        yield action

    def sneak(self, times=1):
        for _ in range(times):
            action = self.create_action()
            action[1] = 1  # Sneak
            yield action

    def sprint(self):
        action = self.create_action()
        action[1] = 2  # Sprint
        yield action

    def attack(self, times=2):
        for _ in range(times):
            action = self.create_action()
            action[5] = 1  # Attack
            yield action

    def use(self):
        action = self.create_action()
        action[6] = 1  # Use
        yield action

    def drop(self):
        action = self.create_action()
        action[7] = 1  # Drop
        yield action

    def craft(self):
        action = self.create_action()
        action[5] = 4  # Craft
        yield action

    def place(self, goal):
        slot = self.index_slot(goal)
        if slot == -1:
            print(f"No slot found for {goal}, skipping place.")
            return
        action = self.create_action()
        action[5] = 6  # Place
        action[7] = slot
        yield action

    def destroy(self):
        action = self.create_action()
        action[5] = 7  # Destroy
        yield action

    def look_horizontal(self, angle_degrees):
        # Constrain angle, convert to range [0..24]
        angle_degrees = max(-20, min(20, angle_degrees))
        normalized = ((angle_degrees + 20) / 40) * 8 + 8
        camera_pos = int(round(normalized))
        camera_pos = max(0, min(24, camera_pos))
        
        action = self.create_action()
        action[4] = camera_pos  # Horizontal look
        yield action

    def no_op(self):
        yield self.env.action_space.no_op()

    ###################################
    # UTILITY
    ###################################
    def is_blocked(self, obs):
        """Example check if movement is blocked by analyzing image brightness near center."""
        height, width, _ = obs["rgb"].shape
        center_y = height // 2
        center_x = width // 2
        region_size = 20
        
        region = obs["rgb"][
            center_y - region_size:center_y + region_size,
            center_x - region_size:center_x + region_size
        ]
        avg_brightness = np.mean(region)
        return avg_brightness < 50  # Arbitrary threshold for 'blocked'

    def acquire_info(self):
        return self.env.step(self.env.action_space.no_op())

    def index_slot(self, goal):
        obs, reward, done, info = self.acquire_info()
        slot = -1
        for item in info.get("inventory", []):
            if item["name"] == goal:
                slot = item["index"]
                break
        return slot

    def equip(self, goal):
        slot = self.index_slot(goal)
        if slot == -1:
            print(f"No slot found for {goal}, cannot equip.")
            return
        action = self.create_action()
        action[5] = 5
        action[7] = slot
        yield action


###################################
# MAIN
###################################
if __name__ == "__main__":
    user_goal_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Explore and gather resources"
    agent = MinecraftAgent(goal_input=user_goal_input)
    agent.run_agent(max_steps=3000)