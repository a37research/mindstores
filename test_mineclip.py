import minedojo
import numpy as np
import torch
import torch.nn.functional as F
from mineclip import MineCLIP
import time

def main():
    # Initialize MineDojo environment with the same config
    print("\nInitializing MineDojo environment...")
    env = minedojo.make(
        task_id="survival",
        image_size=(160, 256)
    )
    
    # Initialize MineCLIP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_cfg = {
        'arch': 'vit_base_p16_fz.v2.t2',
        'hidden_dim': 512,
        'image_feature_dim': 512,
        'mlp_adapter_spec': 'v0-2.t0',
        'pool_type': 'attn.d2.nh8.glusw',
        'resolution': [160, 256]
    }

    print(f"\nInitializing MineCLIP on {device}")
    clip_model = MineCLIP(**clip_cfg)
    clip_model.load_ckpt("attn.pth", strict=True)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Reset environment and get initial observation
    obs = env.reset()
    
    print("\nObservation Space Information:")
    print("RGB Image shape:", obs["rgb"].shape)
    print("Agent Position:", obs["location_stats"]["pos"])
    print("Agent Pitch:", obs["location_stats"]["pitch"])
    print("Agent Yaw:", obs["location_stats"]["yaw"])
    
    # Process image for MineCLIP
    rgb = np.ascontiguousarray(obs['rgb'])
    rgb = torch.from_numpy(rgb).unsqueeze(0)
    rgb = rgb.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    rgb = F.interpolate(rgb, size=(160, 256), mode='bilinear', align_corners=False)
    
    # Get image embeddings
    with torch.no_grad():
        try:
            image_features = clip_model.forward_image(rgb)
            print("\nMineCLIP Image Embedding shape:", image_features.shape)
        except Exception as e:
            print(f"Error getting image embedding: {e}")
            image_features = torch.zeros(1, clip_cfg['image_feature_dim'], device=device)
    
    # Test text encoding
    test_texts = ["a minecraft forest", "a minecraft desert", "a minecraft village"]
    print("\nTesting text encodings for:", test_texts)
    
    with torch.no_grad():
        for text in test_texts:
            try:
                text_features = clip_model.forward_text([text])
                similarity = F.cosine_similarity(image_features, text_features)
                print(f"Similarity with '{text}': {similarity.item():.3f}")
            except Exception as e:
                print(f"Error getting text embedding for '{text}': {e}")
    
    # Take a few random actions and observe the environment
    print("\nTaking a few random actions to observe environment changes...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"\nStep {i+1}:")
        print("Action taken:", action)
        print("New position:", obs["location_stats"]["pos"])
        print("Reward:", reward)
    
    env.close()

if __name__ == "__main__":
    main()
