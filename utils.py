import pandas as pd
import jax.numpy as jnp


def load_data():
    df = pd.read_csv("data/polls.csv")
    df = df.dropna()
    x = df.to_numpy(dtype=int)[:, 5:10]
    x[:, :3] = x[:, :3] - 1
    y = df["bush"].to_numpy()
    return x, y


def load_prev_vote():
    df = pd.read_csv("data/presvote.csv")
    prev_vote = df["g76_84pr"].to_numpy()
    return prev_vote


def prepare_data():
    regions_idx = (
        jnp.array(
            [
                3,
                4,
                4,
                3,
                4,
                4,
                1,
                1,
                5,
                3,
                3,
                4,
                4,
                2,
                2,
                2,
                2,
                3,
                3,
                1,
                1,
                1,
                2,
                2,
                3,
                2,
                4,
                2,
                4,
                1,
                1,
                4,
                1,
                3,
                2,
                2,
                3,
                4,
                1,
                1,
                3,
                2,
                3,
                3,
                4,
                1,
                3,
                4,
                1,
                2,
                4,
            ]
        )
        - 1
    )

    x, y = load_data()
    prev = load_prev_vote()
    state, edu, age, female, black = jnp.split(x, 5, axis=1)
    # beta_array = jnp.array([augment_column.squeeze(), female.squeeze(),
    # black.squeeze(), (female*black).squeeze()]).T
    age_edu = age + 4 * edu
    return {
        "female": female.squeeze(),
        "black": black.squeeze(),
        "y": y,
        "female_black": (female * black).squeeze(),
        "age_edu": age_edu.squeeze(),
        "age": age.squeeze(),
        "edu": edu.squeeze(),
        "age_edu": age_edu.squeeze(),
        "state": state.squeeze(),
        "regions": regions_idx[state].squeeze(),
        "prev": prev[state].squeeze(),
    }
