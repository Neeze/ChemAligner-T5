import lightning as pl
from transformers import AutoModel
from src.backbones.lang.valor_t5 import T5ForConditionalGeneration
from src.backbones.vision.swin import SwinTransformer
import torch
from torch import optim
import math

class T5Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize multimodal text-based model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            args.t5.pretrained_model_name_or_path
        )

        if args.multimodal.use_visual_feature:
            # Initialize visual model
            self.swin_model = SwinTransformer(
                img_size=args.swin.img_size,
                num_classes=0,
                embed_dim=args.swin.embed_dim,
                depths=args.swin.depths,
                num_heads=args.swin.num_heads
            )
        
            self.swin_model.load_state_dict(
                torch.load(args.swin.pretrained_model_path, map_location=device)['encoder']
            )

            if not args.multimodal.trainable_visual:
                self.swin_model.eval()
                for p in self.swin_model.parameters():
                    p.requires_grad = False
        
        
    def resize_token_embeddings(self, len_embeddings):
        self.t5_model.resize_token_embeddings(len_embeddings)
        
    def __prepare_inputs(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        images = batch['images']
        if self.args.multimodal.use_visual_feature:
            if self.args.multimodal.trainable_visual:
                image_features = self.swin_model.forward_features(images, avgpool=False)
            else:
                with torch.no_grad():
                    image_features = self.swin_model.forward_features(images, avgpool=False)
        
        return input_ids, attention_mask, labels, image_features

    def prepare_inputs(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        images = batch['images']
        if self.args.multimodal.use_visual_feature:
            if self.args.multimodal.trainable_visual:
                image_features = self.swin_model.forward_features(images, avgpool=False)
            else:
                with torch.no_grad():
                    image_features = self.swin_model.forward_features(images, avgpool=False)
        
        return input_ids, attention_mask, labels, image_features
    
    def forward(self, input_ids, 
                attention_mask, 
                labels=None,
                image_features=None):
        labels[labels == self.args.tokenizer.pad_token_id] = -100
        
        output = self.t5_model.forward_train(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            image_features=image_features,
            seq2seq_loss_weight=1.0,
            contrastive_loss_weight=0.5,
            output_attentions=True # ADD HERE
        )
        
        return output.loss, output.logits
    
    def forward2(self, input_ids, 
                attention_mask, 
                labels=None, 
                image_features=None):
        labels[labels == self.args.tokenizer.pad_token_id] = -100
        
        output = self.t5_model.forward_train(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            image_features=image_features,
            seq2seq_loss_weight=1.0,
            contrastive_loss_weight=0.5,
            output_attentions=True # ADD HERE
        )
        
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, image_features = self.__prepare_inputs(batch)
        loss, _ = self(input_ids, attention_mask, labels, image_features)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, image_features = self.__prepare_inputs(batch)
        loss, _ = self(input_ids, attention_mask, labels, image_features)
        self.log('eval_loss', loss, prog_bar=True, logger=True)
            
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        
        max_iter = self.args.epochs * self.args.train_data_len
        warmup_steps = int(max_iter * self.args.warmup_ratio)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]
    
    def generate_captioning(self, inputs,
                            max_length = 512,
                            num_beams= 1,
                            do_sample=False,
                            temperature=1.0,
                            decoder_start_token_id=0,
                            eos_token_id=1,
                            pad_token_id=0):
        input_ids, attention_mask, labels = self.__prepare_inputs(inputs)
        outputs = self.t5_model.generate(
            input_ids = input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=decoder_start_token_id,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
        
        outputs = [s.replace('<unk>', '').replace('<pad>', '').replace('</s>', '').strip() for s in self.tokenizer.batch_decode(outputs)]
        
        return outputs
        
    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)