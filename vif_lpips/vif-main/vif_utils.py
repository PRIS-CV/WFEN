import numpy as np


def im2col(img, k, stride=1):
    # Parameters
    m, n = img.shape
    s0, s1 = img.strides
    nrows = m - k + 1
    ncols = n - k + 1
    shape = (k, k, nrows, ncols)
    arr_stride = (s0, s1, s0, s1)

    ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
    return ret[:, :, ::stride, ::stride].reshape(k*k, -1)


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M+1, N+1))
    int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def moments(x, y, k, stride):
    kh = kw = k

    k_norm = k**2

    x_pad = np.pad(x, int((kh - stride)/2), mode='reflect')
    y_pad = np.pad(y, int((kw - stride)/2), mode='reflect')

    int_1_x = integral_image(x_pad)
    int_1_y = integral_image(y_pad)

    int_2_x = integral_image(x_pad*x_pad)
    int_2_y = integral_image(y_pad*y_pad)

    int_xy = integral_image(x_pad*y_pad)

    mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride])/k_norm
    mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] - int_1_y[kh::stride, :-kw:stride] + int_1_y[kh::stride, kw::stride])/k_norm

    var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride])/k_norm - mu_x**2
    var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] - int_2_y[kh::stride, :-kw:stride] + int_2_y[kh::stride, kw::stride])/k_norm - mu_y**2

    cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] - int_xy[kh::stride, :-kw:stride] + int_xy[kh::stride, kw::stride])/k_norm - mu_x*mu_y

    mask_x = (var_x < 0)
    mask_y = (var_y < 0)

    var_x[mask_x] = 0
    var_y[mask_y] = 0

    cov_xy[mask_x + mask_y] = 0

    return (mu_x, mu_y, var_x, var_y, cov_xy)


def vif_gsm_model(pyr, subband_keys, M):
    tol = 1e-15
    s_all = []
    lamda_all = []

    for subband_key in subband_keys:
        y = pyr[subband_key]
        y_size = (int(y.shape[0]/M)*M, int(y.shape[1]/M)*M)
        y = y[:y_size[0], :y_size[1]]

        y_vecs = im2col(y, M, 1)
        cov = np.cov(y_vecs)
        lamda, V = np.linalg.eigh(cov)
        lamda[lamda < tol] = tol
        cov = V@np.diag(lamda)@V.T

        y_vecs = im2col(y, M, M)

        s = np.linalg.inv(cov)@y_vecs
        s = np.sum(s * y_vecs, 0)/(M*M)
        s = s.reshape((int(y_size[0]/M), int(y_size[1]/M)))

        s_all.append(s)
        lamda_all.append(lamda)

    return s_all, lamda_all


def vif_channel_est(pyr_ref, pyr_dist, subband_keys, M):
    tol = 1e-15
    g_all = []
    sigma_vsq_all = []

    for i, subband_key in enumerate(subband_keys):
        y_ref = pyr_ref[subband_key]
        y_dist = pyr_dist[subband_key]

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1

        y_size = (int(y_ref.shape[0]/M)*M, int(y_ref.shape[1]/M)*M)
        y_ref = y_ref[:y_size[0], :y_size[1]]
        y_dist = y_dist[:y_size[0], :y_size[1]]

        mu_x, mu_y, var_x, var_y, cov_xy = moments(y_ref, y_dist, winsize, M)

        g = cov_xy / (var_x + tol)
        sigma_vsq = var_y - g*cov_xy

        g[var_x < tol] = 0
        sigma_vsq[var_x < tol] = var_y[var_x < tol]
        var_x[var_x < tol] = 0

        g[var_y < tol] = 0
        sigma_vsq[var_y < tol] = 0

        sigma_vsq[g < 0] = var_y[g < 0]
        g[g < 0] = 0

        sigma_vsq[sigma_vsq < tol] = tol

        g_all.append(g)
        sigma_vsq_all.append(sigma_vsq)

    return g_all, sigma_vsq_all


def vif(img_ref, img_dist, wavelet='steerable', full=False):
    assert wavelet in ['steerable', 'haar', 'db2', 'bio2.2'], 'Invalid choice of wavelet'
    M = 3
    sigma_nsq = 0.1

    if wavelet == 'steerable':
        from pyrtools.pyramids import SteerablePyramidSpace as SPyr
        pyr_ref = SPyr(img_ref, 4, 5, 'reflect1').pyr_coeffs
        pyr_dist = SPyr(img_dist, 4, 5, 'reflect1').pyr_coeffs
        subband_keys = []
        for key in list(pyr_ref.keys())[1:-2:3]:
            subband_keys.append(key)
    else:
        from pywt import wavedec2
        ret_ref = wavedec2(img_ref, wavelet, 'reflect', 4)
        ret_dist = wavedec2(img_dist, wavelet, 'reflect', 4)
        pyr_ref = {}
        pyr_dist = {}
        subband_keys = []
        for i in range(4):
            pyr_ref[(3-i, 0)] = ret_ref[i+1][0]
            pyr_ref[(3-i, 1)] = ret_ref[i+1][1]
            pyr_dist[(3-i, 0)] = ret_dist[i+1][0]
            pyr_dist[(3-i, 1)] = ret_dist[i+1][1]
            subband_keys.append((3-i, 0))
            subband_keys.append((3-i, 1))
        pyr_ref[4] = ret_ref[0]
        pyr_dist[4] = ret_dist[0]

    subband_keys.reverse()
    n_subbands = len(subband_keys)

    [g_all, sigma_vsq_all] = vif_channel_est(pyr_ref, pyr_dist, subband_keys, M)

    [s_all, lamda_all] = vif_gsm_model(pyr_ref, subband_keys, M)

    nums = np.zeros((n_subbands,))
    dens = np.zeros((n_subbands,))
    for i in range(n_subbands):
        g = g_all[i]
        sigma_vsq = sigma_vsq_all[i]
        s = s_all[i]
        lamda = lamda_all[i]

        n_eigs = len(lamda)

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1
        offset = (winsize - 1)/2
        offset = int(np.ceil(offset/M))

        g = g[offset:-offset, offset:-offset]
        sigma_vsq = sigma_vsq[offset:-offset, offset:-offset]
        s = s[offset:-offset, offset:-offset]

        for j in range(n_eigs):
            nums[i] += np.mean(np.log(1 + g*g*s*lamda[j]/(sigma_vsq+sigma_nsq)))
            dens[i] += np.mean(np.log(1 + s*lamda[j]/sigma_nsq))

    if not full:
        return np.mean(nums + 1e-4)/np.mean(dens + 1e-4)
    else:
        return np.mean(nums + 1e-4)/np.mean(dens + 1e-4), (nums + 1e-4), (dens + 1e-4)


def vif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    mu_x, mu_y, var_x, var_y, cov_xy = moments(x, y, k, stride)

    g = cov_xy / (var_x + 1e-10)
    sv_sq = var_y - g * cov_xy

    g[var_x < 1e-10] = 0
    sv_sq[var_x < 1e-10] = var_y[var_x < 1e-10]
    var_x[var_x < 1e-10] = 0

    g[var_y < 1e-10] = 0
    sv_sq[var_y < 1e-10] = 0

    sv_sq[g < 0] = var_x[g < 0]
    g[g < 0] = 0
    sv_sq[sv_sq < 1e-10] = 1e-10

    vif_val = np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4)
    if (full):
        # vif_map = (np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/(np.log(1 + var_x / sigma_nsq) + 1e-4)
        # return (vif_val, vif_map)
        return (np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4), np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4), vif_val)
    else:
        return vif_val


def msvif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    n_levels = 5
    nums = np.ones((n_levels,))
    dens = np.ones((n_levels,))
    for i in range(n_levels-1):
        if np.min(x.shape) <= k:
            break
        nums[i], dens[i], _ = vif_spatial(x, y, k, sigma_nsq, stride, full=True)
        x = x[:(x.shape[0]//2)*2, :(x.shape[1]//2)*2]
        y = y[:(y.shape[0]//2)*2, :(y.shape[1]//2)*2]
        x = (x[::2, ::2] + x[1::2, ::2] + x[1::2, 1::2] + x[::2, 1::2])/4
        y = (y[::2, ::2] + y[1::2, ::2] + y[1::2, 1::2] + y[::2, 1::2])/4

    if np.min(x.shape) > k:
        nums[-1], dens[-1], _ = vif_spatial(x, y, k, sigma_nsq, stride, full=True)
    msvifval = np.sum(nums) / np.sum(dens)

    if full:
        return msvifval, nums, dens
    else:
        return msvifval
