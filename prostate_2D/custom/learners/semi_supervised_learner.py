from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.apis.dxo import DXO, DataKind, from_shareable

from custom.utils.training_utils import SemanticSeg  # Your monolithic training file

import os
import json
import torch

class SemiSupervisedLearner(Learner, ModelLearner):
    def __init__(self, **config):
        Learner.__init__(self)
        ModelLearner.__init__(self)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_epoch = config.get("n_epoch", 50)
        self.batch_size = config.get("batch_size", 4)
        self.lr = config.get("lr", 1e-4)
        self.channels = config.get("n_channels", 3)
        self.num_classes = config.get("n_classes", 2)
        self.input_shape = config.get("image_size", 384)
        self.split_file = config.get("split_file")
        self.data_root = config.get("data_root")

        self.initialized = False
        self.model = None  # Will be set after training

    def initialize(self, components, fl_ctx):
        site_id = fl_ctx.get_identity_name()

        if not os.path.isdir(self.data_root):
            raise ValueError(f"Invalid or missing data_root: {self.data_root}")

        print(f"[{site_id}] Initializing data from {self.data_root}...")

        with open(self.split_file) as f:
            splits = json.load(f)
        case_ids = splits[site_id]

        self.sample_paths = [os.path.join(self.data_root, cid) for cid in case_ids]
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())  # Get run-specific workspace

        self.log_dir = os.path.join(app_dir, "logs", site_id)
        self.output_dir = os.path.join(app_dir, "checkpoints", site_id)
        self.initialized = True

    def train(self, shareable, fl_ctx, abort_signal):
        if not self.initialized:
            self.initialize(None, fl_ctx)

        site_id = fl_ctx.get_identity_name()
        

        # Split data into train and val sets (simple 90/10 split)
        num_samples = len(self.sample_paths)
        split_idx = int(0.9 * num_samples)
        train_set = self.sample_paths[:split_idx]
        val_set = self.sample_paths[split_idx:]

        seg_model = SemanticSeg(
            lr=self.lr,
            n_epoch=self.n_epoch,
            channels=self.channels,
            num_classes=self.num_classes,
            input_shape=(self.input_shape, self.input_shape),
            batch_size=self.batch_size,
            num_workers=2,
            device="0",  # Assuming GPU 0
            pre_trained=False,
            ckpt_point=False,
            use_fp16=False,
            transformer_depth=18,
            use_transfer_learning=True
        )

        print(f"[{site_id}] Starting training with SemanticSeg...")

        seg_model.trainer(
            train_path=train_set,
            val_path=val_set,
            val_ap=None,
            cur_fold=0,
            output_dir=self.output_dir,
            log_dir=self.log_dir,
            phase="seg"
        )

        # Save trained model for NVFlare
        self.model = seg_model.net

        # return {
        #     "train_dice": seg_model.metrics_threshold
        # }
        # model_dict = self.model.state_dict()
        # model_dict = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        # # Package into DXO
        # dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_dict)
        global_weights = from_shareable(shareable).data
        local_weights = self.model.state_dict()

        # Compute WEIGHT_DIFF
        model_diff = {}
        for k in local_weights:
            if k in global_weights:
                model_diff[k] = local_weights[k].cpu().numpy() - global_weights[k]

        # Package the diff into a DXO with correct DataKind
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)

        dxo.meta["NUM_STEPS_CURRENT_ROUND"] = m
        # Convert to shareable
        return dxo.to_shareable()

    def get_model(self):
        return self.model

    def set_model(self, model):
        if self.model is None:
            print("Model hasn't been trained yet; creating placeholder model.")
            self.model = model  # You may want to reconstruct the same architecture here
        else:
            self.model.load_state_dict(model.state_dict())

    def finalize(self, fl_ctx):
        print("[FL Learner] Finalizing resources.")

