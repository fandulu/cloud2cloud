class Config_cloud():
    def __init__(self):
        self.dataset = 'Cloud_prediction'
        self.data_dir = '/mnt/nasbi/homes/fan/projects/video/cloud_video/'
        self.log_path = "log/" #log dir and saved model dir
        self.model_path = 'save_model/'
        self.checkpoint = self.model_path+'25.pth'
        
        self.DEVICE_ID = "1"
        self.num_workers = 4

        self.max_epoch = 26
        self.save_epoch = 5
        
        self.batch_size = 64
        self.test_batch = 64
        
        self.optimizer = 'Adam'
        self.lr = 1e-5



