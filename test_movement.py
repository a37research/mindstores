import numpy as np
import time
import minedojo

class MovementController:
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
    
    def forward(self, times=100):
        for _ in range(times):
            act = self.env.action_space.no_op()
            act[0] = 1  # Forward
            yield act
    
    def jump_forward(self, times=100):
        for _ in range(times):
            act = self.env.action_space.no_op()
            act[0] = 1  # Forward
            act[2] = 1  # Jump
            yield act
    def backward(self, times=5):
        """Move backward"""
        act = self.env.action_space.no_op()
        act[0] = 2
        yield act

    def move_left(self, times=5):
        """Strafe left"""
        act = self.env.action_space.no_op()
        act[1] = 1
        yield act

    def move_right(self, times=5):
        """Strafe right"""
        act = self.env.action_space.no_op()
        act[1] = 2
        yield act

    def jump(self, times=1):
        """Jump"""
        act = self.env.action_space.no_op()
        act[2] = 1
        yield act

    def sneak(self, times=5):
        """Sneak"""
        act = self.env.action_space.no_op()
        act[2] = 2
        yield act

    def sprint(self, times=5):
        """Sprint"""
        act = self.env.action_space.no_op()
        act[2] = 3
        yield act

    def attack(self, times=3):
        """Attack/break"""
        act = self.env.action_space.no_op()
        act[5] = 3
        yield act

    def use(self):
        """Use/interact"""
        act = self.env.action_space.no_op()
        act[5] = 1
        yield act

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

    def equip(self, slot):
        """Equip from slot"""
        # Set equip action
        act = self.env.action_space.no_op()
        act[5] = 5
        yield act
        
        # Set slot
        act = self.env.action_space.no_op()
        act[6] = max(0, min(35, slot))
        yield act

    def place(self):
        """Place block"""
        act = self.env.action_space.no_op()
        act[5] = 6
        yield act

    def destroy(self):
        """Destroy block"""
        act = self.env.action_space.no_op()
        act[5] = 7
        yield act

def test_movement(env):
    """Test basic movement actions"""
    
    def set_action(env, index, value):
        """Helper to set action array values"""
        action = env.action_space.no_op()
        action[index] = value
        return action
    
    # Test sequence of actions
    actions = [
        ('Forward', lambda: set_action(env, 0, 1)),
        ('Backward', lambda: set_action(env, 0, 2)),
        ('Left', lambda: set_action(env, 1, 1)),
        ('Right', lambda: set_action(env, 1, 2)),
        ('Jump', lambda: set_action(env, 2, 1)),
    ]
    
    for name, action_fn in actions:
        print(f"\nTesting {name} movement...")
        action = action_fn()
        print(f"Action state array: {action}")
        
        # Execute action for a few steps
        for _ in range(5):
            obs, reward, done, info = env.step(action)
            if done:
                print("Episode finished")
                break
                
        # Reset to neutral position
        neutral = env.action_space.no_op()
        print(f"Resetting with action state: {neutral}")
        obs, reward, done, info = env.step(neutral)
        if done:
            print("Episode finished during reset")
            break

# Create the environment
env = minedojo.make(
    task_id="survival",
    image_size=(320, 512)
)

# Reset the environment
obs = env.reset()

# Create controller
controller = MovementController(env)

# Set camera to look straight ahead (index 3 is pitch, 12 is neutral/center)
action = controller.env.action_space.no_op()
action[3] = 12

# First step to set camera position
obs, reward, done, info = env.step(action)
time.sleep(0.2)

# Execute forward jumps
for action in controller.jump_forward(times=100):
    obs, reward, done, info = env.step(action)

for i in range(100):
    test_movement(env)
    controller.jump_forward(env)
env.close()
