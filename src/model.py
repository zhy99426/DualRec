import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .models import DotProductPredictionHead
from .models.dualrec import DualRecModel
from .utils import recalls_and_ndcgs_for_ks, mrr

class DualRecModule(pl.LightningModule):
    def __init__(
        self,
        dualrec: DualRecModel,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        loss_ratio: float = 0.5,
        aux_factor: float = 0.0,
        
    ):
        super().__init__()
        self.dualrec = dualrec
        self.lr = lr
        self.weight_decay = weight_decay
        self.head = DotProductPredictionHead(
            dualrec.d_model, dualrec.num_items, self.dualrec.item_embedding
        )
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.loss_ratio = loss_ratio
        self.aux_factor = aux_factor
        self.epoch = 0
        self.dualrec_reversed = DualRecModel(
            d_model=self.dualrec.d_model,
            d_head=self.dualrec.d_head,
            n_head=self.dualrec.n_head,
            d_inner=self.dualrec.d_inner,
            layer_norm_eps=self.dualrec.layer_norm_eps,
            activation_type=self.dualrec.activation_type,
            clamp_len=self.dualrec.clamp_len,
            n_layer=self.dualrec.n_layer,
            num_items=self.dualrec.num_items,
            seg_len=self.dualrec.seg_len,
            device=self.dualrec.device,
            initializer_range=self.dualrec.initializer_range,
            reverse=True,
            multi_scale = self.dualrec.multi_scale
        )
        self.dualrec_reversed.dropout = self.dualrec.dropout
        self.dualrec_reversed.item_embedding = self.dualrec.item_embedding
        self.dualrec_reversed.pos_embedding = self.dualrec.pos_embedding
        self.dualrec_reversed.mask_emb = self.dualrec.mask_emb

    def forward(self, input_ids1, input_ids2=None):
        outputs = []
        if input_ids2 != None:
            outputs_reversed = []
            for i in range(input_ids1.size(1)):
                input_mask1 = (input_ids1[:, i] == 0).float()
                input_mask2 = (input_ids2[:, i] == 0).float()

                output, attn = self.dualrec(input_ids=input_ids1[:, i], input_mask=input_mask1, output_attentions=True)
                output_reversed, attn_reversed = self.dualrec_reversed(
                    input_ids=input_ids2[:, i], input_mask=input_mask2, output_attentions=True
                )
                outputs.append(output)
                outputs_reversed.append(output_reversed)
                
            outputs = torch.cat(outputs, dim=1)
            outputs_reversed = torch.cat(outputs_reversed, dim=1)
            return outputs, outputs_reversed, attn, attn_reversed

        else:
            for i in range(input_ids1.size(1)):
                input_mask = (input_ids1[:, i] == 0).float()
                output = self.dualrec(input_ids=input_ids1[:, i], input_mask=input_mask)
                outputs.append(output[0])
            outputs = torch.cat(outputs, dim=1)
            return outputs

    def training_step(self, batch, batch_idx):
        input_ids1 = batch["input_ids"][:, :, :-1]
        input_ids2 = batch["input_ids"][:, :, 1:]

        outputs, outputs_reversed, attn, attn_reversed = self(input_ids1, input_ids2)

        mask1 = torch.where(input_ids1.squeeze() != 0)
        mask2 = torch.where(input_ids2.squeeze() != 0)

        logits = self.head(outputs[mask1])  # BT x H

        loss1 = self.loss(logits, input_ids2.squeeze()[mask1]).unsqueeze(-1)
        logits_reversed = self.head(outputs_reversed[mask2])  # BT x H

        loss2 = self.loss(logits_reversed, input_ids1.squeeze()[mask2]).unsqueeze(-1)

        loss = (self.loss_ratio * loss1+ (1-self.loss_ratio)*loss2)
        
        loss += self.aux_factor * (self.compute_kl_loss(attn[0].mean(dim=0).view(-1, self.dualrec.d_head), attn_reversed[0].mean(dim=0).view(-1, self.dualrec.d_head))+
                                    self.compute_kl_loss(attn[1].mean(dim=0).view(-1, self.dualrec.d_head), attn_reversed[1].mean(dim=0).view(-1, self.dualrec.d_head)))/2

        return {"loss": loss}


    def training_epoch_end(self, training_step_outputs):
        loss = torch.cat([o["loss"] for o in training_step_outputs], 0).mean()
        self.log("train_loss", loss)
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        outputs = self(input_ids)

        # get scores (B x C) for evaluation
        last_outputs = outputs[:, -1, :]
        candidates = batch["candidates"].squeeze()  # B x C
        logits = self.head(last_outputs, candidates)  # B x C

        labels = batch["labels"].squeeze()
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])
        metrics["MRR"] = mrr(logits, labels)
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in validation_step_outputs:
                tmp.append(o[k])
            self.log(f"Val:{k}", torch.Tensor(tmp).mean())

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        outputs = self(input_ids)
        # get scores (B x C) for evaluation
        last_outputs = outputs[:, -1, :]
        candidates = batch["candidates"].squeeze()  # B x C
        logits = self.head(last_outputs, candidates)  # B x C

        labels = batch["labels"].squeeze()
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])
        metrics["MRR"] = mrr(logits, labels)
        return metrics

    # def test_step_end(...):
    #     pass

    def test_epoch_end(self, test_step_outputs):
        keys = test_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in test_step_outputs:
                tmp.append(o[k])
            self.log(f"Test:{k}", torch.Tensor(tmp).mean())
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, verbose=True, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler':lr_scheduler, "monitor":"Val:MRR"}
        # return optimizer

    def compute_kl_loss(self, p, q, pad_mask=None):

        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="batchmean"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="batchmean"
        )

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        loss = (p_loss + q_loss) / 2
        return loss
    
    