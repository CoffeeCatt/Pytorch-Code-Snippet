def check_point(epo=None):
    check_pt = {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'np_random_state': prng.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'torch_rng_state_cuda': torch.cuda.get_rng_state()
    }
    if epo is not None:
        torch.save(check_pt, os.path.join(results_folder, '%03d.pt' % epo))
    else:
        torch.save(check_pt, os.path.join(results_folder, 'checkpoint.pt'))

train_from = 'yelp_results/038.pt'

checkpoint = torch.load(train_from, map_location="cuda")

np_random_state = checkpoint['np_random_state']
prng.set_state(np_random_state)
torch_rng_state = checkpoint['torch_rng_state']
torch_rng_state_cuda = checkpoint['torch_rng_state_cuda']
torch.set_rng_state(torch_rng_state.cpu())
torch.cuda.set_rng_state(torch_rng_state_cuda.cpu())

model = checkpoint['model']
criterion = checkpoint['criterion']
optimizer = checkpoint['optimizer']

epo_0 = int(train_from[-6:-3])

for epo in torch.arange(epo_0 + 1, num_epochs + 1):
    print("the current epo is %d of %d" % (epo, num_epochs))
    print("training:")

    # training
    model.train()
    random_bat = torch.randperm(len(train_data)).tolist()
    pbar = tqdm(range(len(train_data)))
    for bat in pbar:
        mini_batch = random_bat[bat]
        sents = train_data[mini_batch]
        
