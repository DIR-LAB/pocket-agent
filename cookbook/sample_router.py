from litellm import Router

router_config = {
    "models": [
        {
            "model_name": "gpt-5-nano",
            "litellm_params": {
                "model": "gpt-5-nano",
                "tpm": 3000000,
                "rpm": 5000
            }
        }
    ]
}

router = Router(model_list=router_config["models"])