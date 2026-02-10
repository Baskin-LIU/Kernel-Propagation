import numpy as np
import torch


class Recorder():

    def __init__(self, names, dt=1.):
        self.record = {name:[] for name in names}
        self.dt = dt

    def rec(self, data):
        for i, rcd in enumerate(self.record.values()):
            rcd.append(data[i].detach().clone())

    def rec_single(self, name, data):
        self.record[name].append(data.detach().clone())
            
    def finish(self,):
        for name, rcd in self.record.items():
            try:
                self.record[name] = torch.vstack(rcd).numpy().T
                self.steps=self.record[name].shape[1]
            except:
                continue

    def __getitem__(self, name):
        # This allows: recorder[name]
        return self.record[name]

def extract_kernel(model, n_steps, layer_idx):
    model.reset()
    impulse = torch.zeros(1, n_steps, 1).to(model.device)
    impulse[0] += 1
    kernel_record = []
    for t in range(n_steps):
        r = impulse[:, t]
        for l in model.layers[:layer_idx]:
            r,u = l.step(r)
        kernel_record.append(u.detach().clone())

    return torch.vstack(kernel_record).cpu().numpy().T
        