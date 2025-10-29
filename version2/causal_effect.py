import numpy as np
import torch
import torch.nn.functional as F
from transformer_vae_cell_rvcse_new import VAE

model = VAE()

params =model.ce_params

def joint_uncond_v1(model, data, alpha_vi=False, beta_vi=True, eps=1e-8, device=None):
    """
    joint_uncond:
        Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
    Inputs:
        - params['N_alpha'] monte-carlo samples per causal factor
        - params['N_beta']  monte-carlo samples per noncausal factor
        - params['K']      number of causal factors
        - params['L']      number of non-causal factors
        - params['M']      number of classes (dimensionality of classifier output)
        - model
        - data
        - device
    Outputs:
        - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
        - info['xhat']
        - info['yhat']
    """
    model.eval()
    params =model.ce_params
    # print(params)
    # exit()
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data.repeat(params['N_alpha'] * params['N_beta'], 1)
    # x1, _ = model.feature_mapper(feat, mode='causal')
    # latent1 = model.encoder1(x1)
    # x2, _ = model.feature_mapper(feat, mode='spurious')
    # latent2 = model.encoder2(x2)
    x, x1,x2,mu_causal,mu_non, log_causal, log_non, cau =model.hidden_pre(feat)
    mu = torch.cat((mu_causal, mu_non), dim=-1)
    log = torch.cat((log_causal, log_non), dim=-1)
    # mu = mu_causal
    # std = log_causal
    # print(mu,std,'std')
    mu = torch.nan_to_num(mu, nan=0.0)
    log = torch.nan_to_num(log, nan=0.0)
    # print(mu,std,'std')
    # mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    std = torch.exp(0.5 * log)
    eps = torch.randn_like(std)
    zs= eps * std + mu
    # print(zs.shape,'zsshape')
    # print(zs,'zs')
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    # logit, prob = model.dpd_model(zs).values()
    prob = model.predic_causal(zs)
    prob = torch.softmax(prob, dim=1)
    # print(prob,"")
    # exit()
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    # if params['M'] == 2:
    #     yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    # else:
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    # yhat = prob.view(params['N_alpha'], params['N_beta'], params['M'])

    p = prob.mean(0)
    # print(prob.shape,'preoshape')
    # print(p.shape,'pshape')
    # p = torch.softmax(logits, dim=1)
    # p = p / (p.sum(dim=1, keepdim=True) + eps)
    eps= torch.randn_like(p)
    # I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=0).mean()
    I = torch.sum(torch.mul(p, F.log_softmax(p, dim=0)), dim=0).mean()

    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    # I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    I = I - torch.sum(torch.mul(q, F.log_softmax(q, dim=0)))
    # exit()
    model.train()
    
    # print(I)
    if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)):
        print("NaN or Inf found in p!")
    if torch.any(torch.isnan(q)) or torch.any(torch.isinf(q)):
        print("NaN or Inf found in q!")
    # exit()
    # print(I)
    return -I


def beta_info_flow_v1( model, data,  alpha_vi=True, beta_vi=False, eps=1e-8, device=None):
    model.eval()
    params =model.ce_params
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data.repeat(params['N_alpha'] * params['N_beta'], 1)

    x, x1,x2,mu_causal,mu_non, log_causal, log_non, cau =model.hidden_pre(feat)

    # mu = mu_causal
    # std = log_causal
    mu = torch.cat((mu_causal, mu_non), dim=-1)
    log = torch.cat((log_causal, log_non), dim=-1)
    mu = torch.nan_to_num(mu, nan=0.0)
    log = torch.nan_to_num(log, nan=0.0)

    std = torch.exp(0.5 * log)
    eps = torch.randn_like(std)
    zs= eps * std + mu
    # if alpha_vi:
    #     alpha_mu = mu[:, :params['K']].mean(0)
    #     alpha_std = std[:, :params['K']].mean(0)
    # else:
    #     alpha_mu = 0
    #     alpha_std = 1

    # if beta_vi:
    #     beta_mu = mu[:, params['K']:].mean(0)
    #     beta_std = std[:, params['K']:].mean(0)
    # else:
    #     beta_mu = 0
    #     beta_std = 1

    # alpha = torch.randn((params['N_alpha'] * params['N_beta'], params['K']), device=device).mul(alpha_std).add_(
    #     alpha_mu)
    # beta = torch.randn((params['N_alpha'], params['L']), device=device).mul(beta_std).add_(beta_mu).repeat(
    #     1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], params['L'])

    # zs = torch.cat([alpha, beta], dim=-1)
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    prob = model.predic_causal(zs)
    prob = torch.softmax(prob, dim=1)
    # print(prob.shape,'preoshape')
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    # if params['M'] == 2:
    #     yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    # else:
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    # yhat = prob.view(params['N_alpha'], params['N_beta'], params['M'])
    p = prob.mean(0)
    # print(p.shape,'pshape')
    eps= torch.randn_like(p)
    # I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=0).mean()
    I = torch.sum(torch.mul(p, F.log_softmax(p, dim=0)), dim=0).mean()

    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    # I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    I = I - torch.sum(torch.mul(q, F.log_softmax(q, dim=0)))
    model.train()
    return -I



def joint_uncond_v2(model, data, alpha_vi=False, beta_vi=True, eps=1e-8, device=None):
    """
    joint_uncond:
        Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
    Inputs:
        - params['N_alpha'] monte-carlo samples per causal factor
        - params['N_beta']  monte-carlo samples per noncausal factor
        - params['K']      number of causal factors
        - params['L']      number of non-causal factors
        - params['M']      number of classes (dimensionality of classifier output)
        - model
        - data
        - device
    Outputs:
        - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
        - info['xhat']
        - info['yhat']
    """
    model.eval()
    params =model.ce_params
    # print(params)
    # exit()
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data.repeat(params['N_alpha'] * params['N_beta'], 1)
    # x1, _ = model.feature_mapper(feat, mode='causal')
    # latent1 = model.encoder1(x1)
    # x2, _ = model.feature_mapper(feat, mode='spurious')
    # latent2 = model.encoder2(x2)
    x, x1,x2,mu_causal,mu_non, log_causal, log_non, cau =model.hidden_pre(feat)
    mu = torch.cat((mu_causal, mu_non), dim=-1)
    std = torch.cat((log_causal.sqrt(), log_non.sqrt()), dim=-1)
    # mu = mu_causal
    # std = log_causal
    # print(mu,std,'std')
    mu = torch.nan_to_num(mu, nan=0.0)
    std = torch.nan_to_num(std, nan=0.0)
    # print(mu,std,'std')
    # mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'], params['K']), device=device).mul(alpha_std).add_(alpha_mu).repeat(1, params[
        'N_beta']).view(params['N_alpha'] * params['N_beta'], params['K'])
    beta = torch.randn((params['N_alpha'] * params['N_beta'], params['L']), device=device).mul(beta_std).add_(beta_mu)
    # exit()
    
    zs = torch.cat([alpha, beta], dim=-1)
    # print(zs,'zs')
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    # logit, prob = model.dpd_model(zs).values()
    prob = model.predic(zs)
    prob = torch.softmax(prob, dim=1)
    # print(prob,"")
    # exit()
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    if params['M'] == 2:
        yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    else:
        yhat = prob.view(params['N_alpha'], params['N_beta'], params['M'])

    p = yhat.mean(1)

    # p = torch.softmax(logits, dim=1)
    # p = p / (p.sum(dim=1, keepdim=True) + eps)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()

    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    # exit()
    model.train()
    return -I


def beta_info_flow_v2( model, data,  alpha_vi=True, beta_vi=False, eps=1e-8, device=None):
    model.eval()
    params =model.ce_params
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data.repeat(params['N_alpha'] * params['N_beta'], 1)

    x, x1,x2,mu_causal,mu_non, log_causal, log_non, cau =model.hidden_pre(feat)

    # mu = mu_causal
    # std = log_causal
    mu = torch.cat((mu_causal, mu_non), dim=-1)
    std = torch.cat((log_causal.sqrt(), log_non.sqrt()), dim=-1)
    mu = torch.nan_to_num(mu, nan=0.0)
    std = torch.nan_to_num(std, nan=0.0)
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'] * params['N_beta'], params['K']), device=device).mul(alpha_std).add_(
        alpha_mu)
    beta = torch.randn((params['N_alpha'], params['L']), device=device).mul(beta_std).add_(beta_mu).repeat(
        1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], params['L'])

    zs = torch.cat([alpha, beta], dim=-1)
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    prob = model.predic(zs)
    prob = torch.softmax(prob, dim=1)
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    if params['M'] == 2:
        yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    else:
        yhat = prob.view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    model.train()
    return -I

