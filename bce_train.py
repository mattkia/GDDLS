import torch
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim

from source.data_loader import DatasetReader
from source.data_transforms import ToTensor
from source.models import vgg16, RiemannianManifoldVGGV2, SaliencyGuidedFilter
##################################################
##################################################
# Configuration
##################################################
add_guided_filter = True
cross_validation = False
##################################################
##################################################
# Initializing the model parameters
##################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_names = ['CAMO-COCO', 'ECSSD', 'PASCAL', 'DUT-OMRON', 'HKU-IS']
dataset_name = dataset_names[4]
batch_size = 1
image_size = 224
lr = 0.0001
bce_epochs = 20
##################################################
##################################################
# Loading the Train and Test datasets
##################################################
print(f'[*] Loading the dataset {dataset_name}...')
train_set = DatasetReader(dataset_name, train=True, mode='edited',
                          transform=transforms.Compose([ToTensor(transpose=False)]))
test_set = DatasetReader(dataset_name, train=False, mode='edited',
                         transform=transforms.Compose([ToTensor(transpose=False)]))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
##################################################
##################################################
# Initializing the model and optimizer
##################################################
vgg = vgg16(pretrained=True).to(device)
model = RiemannianManifoldVGGV2().to(device)
saliency_map = SaliencyGuidedFilter(8, 1e-8)

if not add_guided_filter:
    optimizer = optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = optim.Adam(list(model.parameters()) + list(saliency_map.parameters()), lr=lr)
##################################################
##################################################
# Initializing the loss function
##################################################
bce_loss = torch.nn.BCELoss()
##################################################
##################################################
# Freezing the pre-trained VGG16 network
##################################################
print('[*] Freezing the VGG network')
for parameter in vgg.parameters():
    parameter.requires_grad = False
##################################################
##################################################
# Training the model using Binary Cross Entropy Loss
##################################################
print(f'[*] (BCE Mode) Training the model - Saliency Map : {add_guided_filter} - Cross Validation : {cross_validation}')
model.train()
if add_guided_filter:
    saliency_map.train()

# best_loss is used if cross_validation controller is True
best_loss = 1e5
# Track train loss to break the training procedure in case of abnormality
train_loss = 1e5
for epoch in range(bce_epochs):
    avg_loss = 0.
    start_time = time.time()
    for i, sample in enumerate(train_loader):
        image, label = sample['image'].to(device), sample['label'].to(device)

        optimizer.zero_grad()

        out1 = vgg(image)
        out = model(out1)

        if add_guided_filter:
            out = saliency_map(image, out)

        out = torch.sigmoid(out)

        loss = bce_loss(out, label)

        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(train_set)
    if avg_loss < train_loss:
        train_loss = avg_loss
        if add_guided_filter:
            torch.save(model.state_dict(), 'bce-saved-models/params_model_dgf_8' + dataset_name + '.pt')
            torch.save(saliency_map.state_dict(), 'bce-saved-models/params_dgf_8' + dataset_name + '.pt')
        else:
            torch.save(model.state_dict(), 'bce-saved-models/params_' + dataset_name + '.pt')
        print('[*] BCE trained model was saved successfully!')
    # Cross-Validation of the model (run only if cross_validation is True)
    if cross_validation:
        avg_test_loss = 0.
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                image, label = sample['image'].to(device), sample['label'].to(device)

                out1 = vgg(image)
                out = model(out1)

                if add_guided_filter:
                    out = saliency_map(image, out)

                out = torch.sigmoid(out)

                loss = bce_loss(out, label)

                avg_test_loss += loss.item() / len(test_set)
        print(f'[*] Epoch : {epoch + 1}, Average Train Loss : {avg_loss}, Average Test Loss : {avg_test_loss}, '
              f'Time Elapsed : {(time.time() - start_time) / 60}m')
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            if add_guided_filter:
                torch.save(model.state_dict(), 'bce-saved-models/params_model_dgf_cv_' + dataset_name + '.pt')
                torch.save(saliency_map.state_dict(), 'bce-saved-models/params_dgf_cv_' + dataset_name + '.pt')
            else:
                torch.save(model.state_dict(), 'bce-saved-models/params_cv_' + dataset_name + '.pt')
            print('[*] BCE trained model saved successfully...')
    else:
        print(f'[*] Epoch : {epoch + 1}, Average Loss : {avg_loss}, Time Elapsed : {(time.time() - start_time)/60}m')
##################################################
##################################################
# if not cross_validation:
#     if add_guided_filter:
#         torch.save(model.state_dict(), 'bce-saved-models/params_model_dgf_' + dataset_name + '.pt')
#         torch.save(saliency_map.state_dict(), 'bce-saved-models/params_dgf_' + dataset_name + '.pt')
#     else:
#         torch.save(model.state_dict(), 'bce-saved-models/params_' + dataset_name + '.pt')
#     print('[*] BCE trained model was saved successfully!')
##################################################
##################################################
