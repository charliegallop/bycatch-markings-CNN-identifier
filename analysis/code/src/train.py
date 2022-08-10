from config import DEVICE

# function for running training iterations
def train(self, train_data_loader, model):
    print("-"*50)
    print('Training')
    print("-"*50)
    global train_itr
    global train_loss_all

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total = len(train_data_loader))

    for i, data in enumerate(prog_bar):
        model.train()
        self.optimizer.zero_grad()

        images, labels = data
        #labels = select_labels(LABELS_TO_TRAIN, labels)
        images = list(image.to(DEVICE) for image in images)
        labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
        
        loss_dict = self.model(images, labels)
        summed_losses = sum(loss for loss in loss_dict.values())
        loss_value = summed_losses.item()
        self.train_loss_all.append(loss_value)

        self.train_loss_epoch.send(loss_value)

        summed_losses.backward()
        self.optimizer.step()

        self.train_itr += 1

        # upgrade the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {summed_losses:.4f}")
    return self.train_loss_all