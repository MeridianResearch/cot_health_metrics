# test_wandb.py
try:
    import wandb

    try:
        wandb.init(project="test-project", name="test-run")
        wandb.log({"test_metric": 1.0})
        print(f"✅ W&B test successful! URL: {wandb.run.url}")
        wandb.finish()
    except Exception as e:
        print(f"❌ W&B test failed: {e}")
except ImportError:
    print("Wandb is not installed. Skipping W&B test.")
