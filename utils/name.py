from omegaconf.dictconfig import DictConfig


def get_name_data(config_data: DictConfig) -> str:
    name_list = [config_data.name]
    if config_data.name == 'fastmri':
        name_list.append(f'{config_data.coil}coil')
    elif config_data.name == 'afhq-v2':
        if config_data.num_channels == 1:
            name_list.append('grayscale')
        elif config_data.num_channels == 3:
            name_list.append('color')
        else:
            raise NotImplementedError(f'Unsupported number of channels: {config_data.num_channels}.')
    else:
        raise NotImplementedError(f'Unsupported dataset: {config_data.name}.')
    name_list.append(config_data.group)
    name_list.append(f'{config_data.img_size}x{config_data.img_size}')
    if config_data.normalize:
        name_list.append('normalized')
    return '_'.join(name_list)


def get_name_kernel(config: DictConfig) -> str:
    kernel = config.kernel
    if kernel.name == 'gaussian':
        return '_'.join(filter(None, [kernel.name, f'sigma={kernel.sigma:.1e}']))
    elif kernel.name == 'motion':
        return '_'.join(filter(None, [kernel.name, f'intensity={float(kernel.intensity)}']))
    else:
        raise NotImplementedError(f'Unknown kernel type: {kernel.name}.')


def get_name_operator(operator_config: DictConfig) -> str:

    if operator_config.name == 'single-coil_mri':
        return '_'.join(filter(None, [
            f'{operator_config.mask.name}', 
            f'R={operator_config.mask.acceleration:.1f}', 
            operator_config.mask.orientation, 
            operator_config.mask.pattern,
            f'sigma={operator_config.sigma_noise:.1e}'
        ]))
    elif operator_config.name == 'motion_deblur':
        return '_'.join(filter(None, [
            f'intensity={float(operator_config.intensity)}',
            f'sigma={operator_config.sigma_noise:.1e}',
        ]))
    elif operator_config.name == 'gaussian_deblur':
        return '_'.join(filter(None, [
            f'sigma_kernel={operator_config.sigma_kernel:.1e}',
            f'sigma={operator_config.sigma_noise:.1e}',
        ]))
    elif operator_config.name == 'phase_retrieval':
        return '_'.join(filter(None, [f'oversample={operator_config.oversample:.1f}', f'sigma={operator_config.sigma_noise:.1e}']))
    elif operator_config.name == 'coded_diffraction_pattern':
        return '_'.join(filter(None, [f'{operator_config.num_measurements}measurements', f'noise_std={operator_config.noise_std:.1e}']))
    else:
        raise NotImplementedError(f'Unknown forward model name: {operator_config.name}.')


def get_name_optim(optim: DictConfig) -> str:
    name_list = [
        f'{optim.optimizer}',
        f'bs={optim.batch_size}',
        f'lr={optim.lr:.1e}',
        # f'wd={optim.weight_decay:.1e}',
        f'clip={optim.clip:.1e}'
    ]
    try:
        if optim.ema:
            name_list.append(f'ema_decay={optim.ema_decay:.1e}')
    except:
        pass
    return '_'.join(filter(None, name_list))


def get_name_diffusion(config: DictConfig) -> str:
    model, sde, optim = config.model, config.sde, config.model.optim
    if model.name == 'ncsnpp':
        ch_mult_str = '-'.join(map(str, model.ch_mult))
        attn_res_str = '-'.join(map(str, model.attn_resolutions))
        arch = f'{model.name}_nf={model.nf}_ch={ch_mult_str}_res={model.num_res_blocks}-{model.resblock_type}_attn={attn_res_str}-{model.attention_type}_embd={model.embedding_type}'
    elif model.name == 'ddpm':
        raise NotImplementedError()
    else:
        raise NotImplementedError(f'Unknown model name: {model.name}.')

    if sde.name in ['vp', 'sub_vp']:
        sde = f'{sde.name}_beta={sde.beta_max:.1e}-{sde.beta_min:.1e}' 
    elif sde.name == 've':
        sde = f'{sde.name}_sigma={sde.sigma_max:.1e}-{sde.sigma_min:.1e}'
    else:
        raise NotImplementedError(f'Unknown SDE name: {sde.name}.')

    optim = get_name_optim(optim) + ('_llh-weighting' if optim.llh_weighting else '')

    return '_'.join(filter(None, [arch, sde, optim]))


def get_name_diffusion_new(config: DictConfig) -> str:
    model_config, loss_config, optim_config = config.model_config, config.loss_config, config.optim_config
    if model_config.architecture in ['ncsnpp', 'adm']:
        ch_mult_str = '-'.join(map(str, model_config.channel_mult))
        attn_res_str = '-'.join(map(str, model_config.attn_resolutions))
        arch = '-'.join(filter(None, [
            loss_config.name,
            model_config.architecture, 
            f'ch={model_config.model_channels}-{ch_mult_str}', 
            f'res={model_config.num_res_blocks}', 
            f'attn={attn_res_str}', 
            f'dropout={model_config.dropout:.1f}', 
        ]))
    else:
        raise NotImplementedError(f'Unknown model name: {model_config.name}.')


    optim = get_name_optim(optim_config)

    return '_'.join(filter(None, [arch, optim]))


def get_name_sampler(config_sampler: DictConfig) -> str:
    if config_sampler.name == 'diffusion':
        return "_".join(filter(None, [
            f'schedule={config_sampler.diffusion_scheduler_config.schedule}',
            f'timestep={config_sampler.diffusion_scheduler_config.timestep}',
            f'scaling={config_sampler.diffusion_scheduler_config.scaling}',
            f'sigma={config_sampler.diffusion_scheduler_config.sigma_max:.1e}-{config_sampler.diffusion_scheduler_config.sigma_min:.1e}',
            f'steps={config_sampler.diffusion_scheduler_config.num_steps}',
        ]))
    elif config_sampler.name == 'dps':
        return "_".join(filter(None, [
            f'schedule={config_sampler.diffusion_scheduler_config.schedule}',
            f'timestep={config_sampler.diffusion_scheduler_config.timestep}',
            f'scaling={config_sampler.diffusion_scheduler_config.scaling}',
            f'sigma={config_sampler.diffusion_scheduler_config.sigma_max:.1e}-{config_sampler.diffusion_scheduler_config.sigma_min:.1e}',
            f'steps={config_sampler.diffusion_scheduler_config.num_steps}',
            f'zeta={config_sampler.zeta:.1e}',
        ]))
    elif config_sampler.name == 'score-mri':
        return "_".join(filter(None, [
            f'schedule={config_sampler.scheduler_config.schedule}',
            f'timestep={config_sampler.scheduler_config.timestep}',
            f'scaling={config_sampler.scheduler_config.scaling}',
            f'steps={config_sampler.scheduler_config.num_steps}',
            f'correct_steps={config_sampler.correct_steps}',
            f'lr={config_sampler.lr:.1e}',
            f'lamb={config_sampler.lamb:.1e}',
        ]))
    elif config_sampler.name == 'score-ald':
        return "_".join(filter(None, [
            config_sampler.sigmas_config.sigma_dist,
            f'sigma={config_sampler.sigmas_config.sigma_begin:.1e}-{config_sampler.sigmas_config.sigma_end:.1e}',
            f'steps={config_sampler.sigmas_config.num_steps}',
            f'start_iter={config_sampler.start_iter}',
            f'n_steps_each={config_sampler.n_steps_each}',
            f'lr={config_sampler.step_lr:.1e}',
            f'mse={config_sampler.mse:.1e}',
        ]))
    elif config_sampler.name == 'ald-tv':
        return "_".join(filter(None, [
            f'weight={config_sampler.tv_config.weight:.1e}',
            f'mode={config_sampler.tv_config.mode}',
            config_sampler.sigmas_config.sigma_dist,
            f'sigma={config_sampler.sigmas_config.sigma_begin:.1e}-{config_sampler.sigmas_config.sigma_end:.1e}',
            f'steps={config_sampler.sigmas_config.num_steps}',
            f'start_iter={config_sampler.start_iter}',
            f'n_steps_each={config_sampler.n_steps_each}',
            f'lr={config_sampler.step_lr:.1e}',
            f'mse={config_sampler.mse:.1e}',
        ]))
    elif config_sampler.name == 'rml-tv':
        return "_".join(filter(None, [
            f'weight={config_sampler.tv_config.weight:.1e}',
            f'mode={config_sampler.tv_config.mode}',
            f'steps={config_sampler.num_steps}',
            f'lr={config_sampler.lr:.1e}',
        ]))
    elif config_sampler.name == 'wiener':
        return "_".join(filter(None, [
            f'lam={config_sampler.lam:.1e}',
        ]))
    elif config_sampler.name == 'red-diff':
        return "_".join(filter(None, [
            f'meas_weight={config_sampler.measurement_weight:.1e}',
            f'base_lr={config_sampler.base_lr:.1e}',
            f'base_lambda={config_sampler.base_lambda:.1e}',
            f'lambda_schedule={config_sampler.lambda_schedule}',
            f'steps={config_sampler.num_steps}',
        ]))
    elif config_sampler.name == 'daps':
        annealing, diffusion, lgvd = config_sampler.annealing_scheduler_config, config_sampler.diffusion_scheduler_config, config_sampler.lgvd_config
        return '_'.join(filter(None, [
            'annealing',
            f'schedule={annealing.schedule}',
            f'timestep={annealing.timestep}',
            f'sigma={annealing.sigma_max:.1e}-{annealing.sigma_min:.1e}-{annealing.sigma_final:.1e}',
            f'steps={annealing.num_steps}',
            'diffusion',
            f'schedule={diffusion.schedule}',
            f'timestep={diffusion.timestep}',
            f'sigma={diffusion.sigma_min:.1e}-{diffusion.sigma_final:.1e}',
            f'steps={diffusion.num_steps}',
            'lgvd',
            f'lr={lgvd.lr:.1e}',
            f'lr_min_ratio={lgvd.lr_min_ratio:.1e}',
            f'tau={lgvd.tau:.1e}',
            f'steps={lgvd.num_steps}',
        ]))
    elif config_sampler.name == 'pnp-dm':
        annealing, diffusion, lgvd = config_sampler.annealing_scheduler_config, config_sampler.diffusion_scheduler_config, config_sampler.lgvd_config
        return '_'.join(filter(None, [
            'annealing',
            f'rho={annealing.rho_max:.1e}-{annealing.rho_min:.1e}-{annealing.alpha:.1e}',
            f'steps={annealing.num_steps}',
            'diffusion',
            f'schedule={diffusion.schedule}',
            f'timestep={diffusion.timestep}',
            f'sigma={diffusion.sigma_min:.1e}-{diffusion.sigma_final:.1e}',
            f'steps={diffusion.num_steps}',
            'lgvd',
            f'lr={lgvd.lr:.1e}',
            f'tau={lgvd.tau:.1e}',
            f'steps={lgvd.num_steps}',
        ]))
    else:
        raise NotImplementedError(f'Unsupported sampler: {config_sampler.name}.')


def get_name_nf(nf: DictConfig) -> str:
    return f'{nf.els}els_{nf.efs}efs_{nf.act}' + (f'_{nf.n_freqs}freqs' if nf.pos_enc else '')


def get_name_vi_loss(loss: DictConfig, prior: DictConfig) -> str:
    name_list = []
    
    if loss.alpha == 1.0:
        name_list.append('kl')
    else:
        name_list.append(f'alpha={loss.alpha:.1e}')
        
    if loss.sigma:
        name_list.append(f'sigma={loss.sigma:.1e}')
        
    name_list.append(f'{prior.name}_lam={float(prior.weight):.1e}')
    if prior.name == 'l2-score':
        name_list.append(f'sigma={prior.sigma:.1e}')
        
    name_list.append(f'beta={loss.beta:.1e}')
    
    if loss.anneal:
        name_list.append(f'anneal={loss.anneal}-{loss.starting_order:.1f}-{loss.decay_rate:.1e}')
    
    return '_'.join(name_list)


def get_name_flow(flow: DictConfig) -> str:
    if 'shuffleflow' in flow.name:
        name_list = [
            flow.network_config.name,
            f'downsample={flow.network_config.downsample:.0f}',
            f'flows={flow.network_config.base_flows}-{flow.network_config.shuffle_flows}-{flow.network_config.patch_flows}',
            f'embed={flow.network_config.embed_dim}',
            f'permute={flow.network_config.permute}',
            f'els={flow.network_config.encoder.hidden_layers}',
            f'efs={flow.network_config.encoder.hidden_features}',
            f'act={flow.network_config.encoder.act_func}',
            f'pos_enc={flow.network_config.encoder.num_freqs if flow.network_config.encoder.pos_enc else 0}',
            f'seqfrac={flow.network_config.decoder.seqfrac:.1e}',
            f'batch_norm={flow.network_config.decoder.batch_norm}',
        ]
    elif flow.name == 'patchflow':
        name_list = [
            flow.network_config.name,
            f'downsample={flow.network_config.downsample:.0f}',
            f'flows={flow.network_config.base_flows}-{flow.network_config.patch_flows}',
            f'embed={flow.network_config.embed_dim}',
            f'permute={flow.network_config.permute}',
            f'els={flow.network_config.encoder.hidden_layers}',
            f'efs={flow.network_config.encoder.hidden_features}',
            f'act={flow.network_config.encoder.act_func}',
            f'pos_enc={flow.network_config.encoder.num_freqs if flow.network_config.encoder.pos_enc else 0}',
            f'seqfrac={flow.network_config.decoder.seqfrac:.1e}',
            f'batch_norm={flow.network_config.decoder.batch_norm}',
        ]
    elif flow.name == 'dpi':
        name_list = [
            flow.network_config.name,
            f'flows={flow.network_config.num_flows}',
            f'permute={flow.network_config.permute}',
            f'seqfrac={flow.network_config.seqfrac:.1e}',
            f'batch_norm={flow.network_config.batch_norm}',
        ]
    elif flow.name == 'cf-nerf':
        name_list = [
            flow.network_config.flow_func,
            f'flows={flow.network_config.num_flows}',
            f'embed={flow.network_config.embed_dim}',
            f'els={flow.network_config.encoder.hidden_layers}',
            f'efs={flow.network_config.encoder.hidden_features}',
            f'act={flow.network_config.encoder.act_func}',
            f'pos_enc={flow.network_config.encoder.pos_enc}',
            f'pos_enc_freqs={flow.network_config.encoder.num_freqs}',
            f'dls={flow.network_config.decoder.hidden_layers}',
            f'dfs={flow.network_config.decoder.hidden_features}',
            f'batch_norm={flow.network_config.decoder.batch_norm}',
        ]
    elif flow.name in ['nf_gaussian', 'pg_gaussian']:
        name_list = []
    else:
        raise ValueError(f'VI family {flow.name} not supported.')
    
    if flow.network_config.sigmoid:
        name_list.append('sigmoid')
    
    return '_'.join(name_list)


def get_name_vi(config: DictConfig) -> str:
    flow = config.flow
    prior = config.prior
    name_list = []
    
    # Flow parameters.
    name_list.append(get_name_flow(flow))
    
    # Neural field parameters.
    if flow.name in ['patch_flow', 'pixel_flow', 'nf_gaussian']:
        name_list.append(get_name_nf(flow.nf))
    
    # Loss parameters.
    name_list.append(get_name_vi_loss(flow.loss_config, prior))
    
    # Optimization parameters.
    name_list.append(get_name_optim(flow.optim_config))
    
    return '_'.join(name_list)
