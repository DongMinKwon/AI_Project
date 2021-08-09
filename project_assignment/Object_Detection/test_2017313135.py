from train_2017313135 import *

def test(test_loader):
    print("\n Start testing")
    print("------------------------------------------------\n\n")
    
    model = Obj_Detect()
    model.load_state_dict(torch.load('./model_2017313135.pt'))
    model = model.to(device)
    
    total_loss = 0
    total_score_loss = 0
    total_test_batch = 0

    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img, target = img.to(device), target.to(device)
            total_test_batch += target.size(0)

            results = model(img)
            loss, confidence = loss_func(results, target)

            total_loss += loss.item()
            total_score_loss += confidence.item()
    print('TEST RESULT : Total_Loss: {:.4f}, avg_confidence_score_loss: {:.5f}'.format(total_loss, total_score_loss/total_test_batch))