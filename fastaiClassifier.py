from fastai.conv_learner import *
from fastai.models.cifar10.senet import SENet18
import mlinstrumentation

# init/start instrumentation
mlinstrumentation.start()


# define your features in dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))

# read your dataset
PATH = Path("data/cifar10/")
os.makedirs(PATH,exist_ok=True)

bs=32
sz=32

tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomCrop(sz), RandomFlip()], pad=sz//8)
data = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)

# define model
m = SENet18()
learn = ConvLearner.from_model_data(m, data)
learn.crit = nn.CrossEntropyLoss()
learn.metrics = [accuracy]
wd=1e-4
lr=1.5

# train model
learn.fit(lr, 1, wds=wd, cycle_len=1)

# deinit/stop instrumentation
mlinstrumentation.stop()
