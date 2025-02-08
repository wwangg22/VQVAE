from model import VQVAE
import torch
from mss import mss
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = VQVAE(h_dim=512, n_embeddings=16, embedding_dim=16).to(device)
model = VQVAE(h_dim=512, n_embeddings=32, embedding_dim=32, scale_factor=6).to(device)
model.load_model("model.pth")

model.eval()
sct = mss()
monitor = {
    "top": 50,     
    "left": 0,      
    "width": 720,   
    "height": 1300  
}

def main():
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)
        
        # Check if the frame was captured correctly
    if frame.size == 0:
        print("Error: Captured frame is empty. Check monitor coordinates and ensure the target window is visible.")
        
    # Convert from BGRA to BGR (OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    frame = cv2.resize(frame,(256,256))

    frame = frame.transpose(2, 0, 1)

    frame = frame.astype(np.float32) / 255.0
    frame = torch.from_numpy(frame).unsqueeze(0).to(device)

    with torch.no_grad():

        _, x_hat, _ = model(frame)

        x_hat = x_hat[0].detach().cpu().numpy()

        x_hat_img = np.transpose(x_hat,(1,2,0))

        cv2.imshow('frame', x_hat_img)

if __name__=="__main__":
    while True:
        main()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


