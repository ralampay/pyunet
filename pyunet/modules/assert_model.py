import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import initialize_model

class AssertModel:
    def __init__(
        self, 
        model_type, 
        in_channels, 
        out_channels, 
        img_width, 
        img_height, 
        device='cuda',
        gpu_index=0
    ):
        self.model_type     = model_type
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.img_width      = img_width
        self.img_height     = img_height
        self.device         = cuda
        self.gpu_index      = gpu_index

    def execute(self):
        print(f"Asserting model {self.model_type}")

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        self.model = initialize_model(
            self.in_channels,
            self.out_channels,
            self.model_type,
            self.device
        )
       
        tensors = torch.randn(
            1, 
            self.in_channels, 
            self.img_width, 
            self.img_height
        ).to(self.device)

        print(f"Shape of tensors: {tensors.shape}")
        print(f"Datatype of tensors: {tensors.dtype}")
        print(f"Device tensors is stored on: {tensors.device}")

        result = self.model(tensors)

        print(result.shape)

        print("Done")
