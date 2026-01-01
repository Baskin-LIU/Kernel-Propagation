import torch

def buildMNISTNet(model_config, general_config)

    LP_size = list(model_config['LP_size'])
    Ins_size = list(model_config['Ins_size'])
    dt = general_config['dt']
    tau = []
    tau_min, tau_max = model_config['Tau0']
    #tau.append(np.logspace(np.log(tau_min), np.log(tau_max), LP_size[0], dtype=np.float32))
    tau.append(np.linspace(tau_min, tau_max, LP_size[0], dtype=np.float32))
    for i in range(model_config['num_LP_layers']-1):
        tau_uniq = model_config['Tau%d'%(i+1)]
        tau.append(torch.tensor(np.repeat(tau_uniq[:, None],
                                          LP_size[i+1]//tau_uniq.shape[0])))
    
    layers = torch.nn.ModuleList()
    prev_n=model_config['n_in']
    
    for i in range(model_config['num_LP_layers']-1):
        scale = 1. if i==0 else 0.2
        layers.append(
            FwdDENeurons(
                n_in=prev_n,
                n_neurons=LP_size[i],
                tau=tau[i], 
                activation=model_config["activation"], 
                dt=dt, 
                scale=0.2
                )
        )
        prev_n=LP_size[i]

    layers.append(
        LastFwdDENeurons(
            n_in=prev_n, 
            n_neurons=LP_size[-1], 
            tau=tau[i+1], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.6
            )
    )
    prev_n=LP_size[-1]
    for i in range(model_config['num_Ins_layers']):
        layers.append(
            FwdInsNeurons(
                n_in=prev_n,
                n_neurons=Ins_size[i],
                activation=model_config["activation"], 
                scale=1.0,
                dt=dt
            )
        )
        prev_n=Ins_size[i]
                
    layers.append(
        FwdInsNeurons(
            n_in=prev_n, 
            n_neurons=model_config['n_out'],
            activation='linear', 
            dt=dt, 
            scale=1.0
            )
    )

    
    return DENetwork(layers=layers,)


def train_batch(model, x, y):
    batch, n_steps, _ = x.shape
    model.reset(batch)
    total_error = 0.
    one_hot_label = torch.zeros(batch, 10)   
    one_hot_label[np.arange(batch), y[idx]] = 1.
    with torch.no_grad():
        for t in range(n_steps):
            r_out, _= model.step(x[:, t])
            if t>=136:
                p = torch.softmax(r_out, dim=1)
                error = one_hot_label - p
                model.prop(error)
                model.backwards()
                optimizer.step()
                optimizer.zero_grad()
                total_error += -(one_hot_label * torch.log(p)).mean().item()
            else:
                model.prop(0)
    return total_error


def test(model, x, y):
    prediction = []
    batch, n_steps, _ = x.shape
    acc = 0.
    model.reset()
    with torch.no_grad():
        pred = torch.zeros(batch, 10)
        for t in range(n_steps):
            r_out,_= model.step(x[:, t])
            if t>130:
                pred += r_out
        prediction=torch.argmax(pred, dim=1)
    acc += ((prediction.numpy()==y[idx])*1.).mean()
    return acc