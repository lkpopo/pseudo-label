from torch.utils.data import DataLoader
import torch
from scipy.special import softmax
from tqdm import tqdm
from train import generate_transform,initialize_model,PlantDataset,load_data
def gengerate_submission(model_name):
    PATH = [
        "./logs_submit/1fold.ckpt",
        "./logs_submit/2fold.ckpt",
        "./logs_submit/apple.ckpt",
        "./logs_submit/4fold.ckpt",
        "./logs_submit/5fold.ckpt",
    ]
    tf = generate_transform()
    data = load_data('apple_test')
    test_data = PlantDataset(data, 'apple_test', tf['train_transforms'])
    test_dataloader = DataLoader(
        test_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, drop_last=False,
    )
    model = initialize_model(model_name)
    submission = []
    for path in PATH:
        model.load_state_dict(torch.load(path))
        model.to("cuda:1")
        model.eval()

        for i in range(2):
            test_preds = []
            labels = []
            with torch.no_grad():
                for image, label, times in tqdm(test_dataloader):
                    test_preds.append(model(image.to("cuda:1")))
                    labels.append(label)
                test_preds = torch.cat(test_preds)
                submission.append(test_preds.cpu().numpy())

    print(submission)
    submission_ensembled = 0
    for sub in submission:
        submission_ensembled += softmax(sub, axis=1) / len(submission)
    print('------', submission_ensembled)
    print(test_data.iloc[:, 1:])
    test_data.iloc[:, 1:] = submission_ensembled
    test_data.to_csv("submission.csv", index=False)
    test_data.to_csv("submission.csv", index=False)