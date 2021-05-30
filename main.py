import os
import argparse
import torch
import model
from dataloader import DataLoader
from trainer import Trainer
import attention_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments
    # Hyperparameters
    parser.add_argument("--num_epochs", default=20, type=int, help="The num of epochs for training")
    parser.add_argument("--hidden_dim", default=256, type=int, help="hidden dim size")
    parser.add_argument("--batch_size", default=64, type=int, help="The num of batch size")
    parser.add_argument("--device", default= torch.device("cuda"), type=str, help="device type")
    parser.add_argument("--max_vocab", default=9999999, type=int, help="max_vocab")
    parser.add_argument("--max_length", default=255, type=int, help="max_length")
    parser.add_argument("--use_attention", default="None", type=str, help="use attention")
    args = parser.parse_args()
  

    loaders = DataLoader(args.batch_size, args.device)
    input_dim = len(loaders.src.vocab)
    output_dim = len(loaders.trg.vocab)
    enc_emb_dim = 256
    dec_emb_dim = 256
    if args.use_attention=='None':
        enc =model.Encoder(input_dim, enc_emb_dim, args.hidden_dim)
        dec = model.Decoder( output_dim,dec_emb_dim,args.hidden_dim) 
        use_model = model.Seq2Seq(enc, dec, args.device) 
        
    else:
        enc =model.Encoder(input_dim, enc_emb_dim, args.hidden_dim)
        dec = attention_model.attn_Decoder( dec_emb_dim,args.hidden_dim,output_dim)
        use_model = attention_model.attn_Seq2Seq(enc, dec, args.device)
        
    
    trainer = Trainer(args, loaders, use_model)

    print("-------Train Start------")
    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train_loss = trainer.train(loaders.train_iterator)
        valid_loss = trainer.evaluate(loaders.valid_iterator)
        print("Epoch[{}/{}], Train Loss : {:.4f}, Valid Loss : {:.4f}".format(epoch + 1, args.num_epochs, train_loss, valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(use_model.state_dict(), "epoch_{}.pth".format( epoch+1))

    print("-------Train Ended------")

    # load best epoch model and evaluate on test set
    use_model.load_state_dict(torch.load('./epoch_{}.pth'.format(best_epoch)))
    test_loss= trainer.evaluate(loaders.test_iterator)
    print("\n[Using Epoch {}'s model, evaluate on Test set]".format(best_epoch))
    print("Test Loss:{:.4f}".format(test_loss))

    


