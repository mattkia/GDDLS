import torch
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim

from source.data_loader import DatasetReader
from source.data_transforms import ToTensor
from source.utils import level_set_loss_v2, normal_vector_loss, curvature_loss
from source.utils import make_dx, make_dy, make_dxx, make_dxy, make_dyy
from source.models import vgg16, RiemannianManifoldVGGV2, SaliencyGuidedFilter
##################################################
##################################################
# Configuration
##################################################
add_guided_filter = True
cross_validation = False
bce_pretrained = True
use_best_train_loss = True
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
epochs = 15
##################################################
# Creating derivative matrices
##################################################
dx = make_dx(image_size)
dy = make_dy(image_size)
dxx = make_dxx(image_size)
dxy, dyx = make_dxy(image_size)
dyy = make_dyy(image_size)
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
# Freezing the pre-trained VGG16 network
##################################################
print('[*] Freezing the VGG network')
for parameter in vgg.parameters():
    parameter.requires_grad = False
##################################################
##################################################
# Training the model
##################################################
if bce_pretrained:
    print(f'[*] Training the BCE pre-trained model - Saliency Map : {add_guided_filter} - '
          f'Cross Validation : {cross_validation}')
else:
    print(f'[*] Training the model - Saliency Map : {add_guided_filter} - Cross Validation : {cross_validation}')
# Loading the BCE pre-trained model (if bce_pretrained is True)
if bce_pretrained:
    if cross_validation:
        if add_guided_filter:
            model.load_state_dict(torch.load('bce-saved-models/params_model_dgf_cv_' + dataset_name + '.pt'))
            saliency_map.load_state_dict(torch.load('bce-saved-models/params_dgf_cv_' + dataset_name + '.pt'))
        else:
            model.load_state_dict(torch.load('bce-saved-models/params_cv_' + dataset_name + '.pt'))
    else:
        if add_guided_filter:
            model.load_state_dict(torch.load('bce-saved-models/params_model_dgf_8' + dataset_name + '.pt'))
            saliency_map.load_state_dict(torch.load('bce-saved-models/params_dgf_8' + dataset_name + '.pt'))
        else:
            model.load_state_dict(torch.load('bce-saved-models/params_' + dataset_name + '.pt'))
##################################################
model.train()
if add_guided_filter:
    saliency_map.train()
# best_loss is used only if the cross_validation controller is True
best_loss = 1e5
best_train_loss = 1e5
for epoch in range(epochs):
    avg_loss = 0.
    start_time = time.time()
    for i, sample in enumerate(train_loader):
        image, label = sample['image'].to(device), sample['label'].to(device)

        optimizer.zero_grad()

        out1 = vgg(image)
        out = model(out1)

        if add_guided_filter:
            out = saliency_map(image, out)

        loss1 = level_set_loss_v2(out, label)
        loss2 = normal_vector_loss(out, dx, dy)
        loss3 = curvature_loss(out, dx, dy, dxx, dxy, dyx, dyy)

        loss = loss1 + 0.3 * loss2 + 0.1 * loss3

        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(train_set)
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

                loss1 = level_set_loss_v2(out, label)
                loss2 = normal_vector_loss(out, dx, dy)
                loss3 = curvature_loss(out, dx, dy, dxx, dxy, dyx, dyy)

                loss = loss1 + 0.3 * loss2 + 0.1 * loss3

                avg_test_loss += loss.item() / len(test_set)
        print(f'[*] Epoch : {epoch + 1}, Average Train Loss : {avg_loss}, Average Test Loss : {avg_test_loss}, '
              f'Time Elapsed : {(time.time() - start_time)/60}m')
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            if bce_pretrained:
                if add_guided_filter:
                    torch.save(model.state_dict(), 'saved-models/params_model_dgf_cv_bce_' + dataset_name + '.pt')
                    torch.save(saliency_map.state_dict(), 'saved-models/params_dgf_cv_bce_' + dataset_name + '.pt')
                else:
                    torch.save(model.state_dict(), 'saved-models/params_cv_bce_' + dataset_name + '.pt')
                print('[*] BCE pre-trained model saved successfully...')
            else:
                if add_guided_filter:
                    torch.save(model.state_dict(), 'saved-models/params_model_dgf_cv_' + dataset_name + '.pt')
                    torch.save(saliency_map.state_dict(), 'saved-models/params_dgf_cv_' + dataset_name + '.pt')
                else:
                    torch.save(model.state_dict(), 'saved-models/params_cv_' + dataset_name + '.pt')
                print('[*] Model saved successfully...')
    else:
        if use_best_train_loss:
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                if bce_pretrained:
                    if add_guided_filter:
                        torch.save(model.state_dict(), 'saved-models/params_model_dgf_bce_8' + dataset_name + '.pt')
                        torch.save(saliency_map.state_dict(), 'saved-models/params_dgf_bce_8' + dataset_name + '.pt')
                    else:
                        torch.save(model.state_dict(), 'saved-models/params_bce_' + dataset_name + '.pt')
                    print('[*] BCE pre-trained model was saved successfully!')
                else:
                    if add_guided_filter:
                        torch.save(model.state_dict(), 'saved-models/params_model_dgf_' + dataset_name + '.pt')
                        torch.save(saliency_map.state_dict(), 'saved-models/params_dgf_' + dataset_name + '.pt')
                    else:
                        torch.save(model.state_dict(), 'saved-models/params_' + dataset_name + '.pt')
                    print('[*] Model was saved successfully!')

        print(f'[*] Epoch : {epoch + 1}, Average Loss : {avg_loss}, Time Elapsed : {(time.time() - start_time)/60}m')
##################################################
##################################################
if not cross_validation and not use_best_train_loss:
    if bce_pretrained:
        if add_guided_filter:
            torch.save(model.state_dict(), 'saved-models/params_model_dgf_bce_' + dataset_name + '.pt')
            torch.save(saliency_map.state_dict(), 'saved-models/params_dgf_bce_' + dataset_name + '.pt')
        else:
            torch.save(model.state_dict(), 'saved-models/params_bce_' + dataset_name + '.pt')
        print('[*] BCE pre-trained model was saved successfully!')
    else:
        if add_guided_filter:
            torch.save(model.state_dict(), 'saved-models/params_model_dgf_' + dataset_name + '.pt')
            torch.save(saliency_map.state_dict(), 'saved-models/params_dgf_' + dataset_name + '.pt')
        else:
            torch.save(model.state_dict(), 'saved-models/params_' + dataset_name + '.pt')
        print('[*] Model was saved successfully!')
##################################################
##################################################
