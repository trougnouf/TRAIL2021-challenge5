from tensorboard_reducer import load_tb_events, reduce_events, write_tb_events

out_dir = "runs/baseline_pretrained_supervised"
overwrite = False
reduce_ops = ("mean",)  # "mean", "min", "max", "std"

events_dict = load_tb_events("runs/sup_pretrained*", strict_steps=False)

n_scalars = len(events_dict)
n_steps, n_events = list(events_dict.values())[0].shape

print(f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars and {n_steps} steps each")
print(", ".join(events_dict))

reduced_events = reduce_events(events_dict, reduce_ops)

for op in reduce_ops:
    print(f"Writing '{op}' reduction to '{out_dir}-{op}'")

write_tb_events(reduced_events, out_dir, overwrite)


print("Reduction complete")
