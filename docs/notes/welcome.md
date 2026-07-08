This site used to run on Jekyll, with a CMS, a card game, and a decade of accumulated
scaffolding. It grew hard to touch, so I stopped touching it. This is the rebuild: a
plain static page with no build step, meant to be simple enough that keeping it alive is
never the reason I stop writing.

This note doubles as a test of the notebook — if the math and code below render, everything works.

## Writing math

Inline math like $e^{i\pi} + 1 = 0$ sits in the middle of a sentence. Display math gets
its own line:

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t),
\qquad
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\big(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\big).
$$

The reverse process of a diffusion model, in one line, for when I inevitably write about it here.

## Writing code

```python
def elbo(model, x0):
    t = sample_timesteps(x0.shape[0])
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)
    return F.mse_loss(model(xt, t), noise)
```

## Adding a note

1. Drop a `.md` file in `notes/` — for example `notes/my-post.md`.
2. Add one entry to `notes/index.json`:

   ```json
   { "slug": "my-post", "title": "My post", "date": "2026", "summary": "One line." }
   ```

3. That's it. The post lives at `note.html?slug=my-post` and shows up on the home page
   and in the notes list automatically.

No server, no rebuild, no CMS. Just markdown.
