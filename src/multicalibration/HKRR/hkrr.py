import numpy as np
from tqdm import trange


class HKRRAlgorithm:
    """
    HKRR, Algorithm 1. Based partially on the code here:
    https://github.com/sanatonek/fairness-and-callibration/tree/893c9738bf8e01d089568b1d7a56a8b53037e5fb
    """
    def __init__(self, verbose=False):
        """
        Multicalibrate Predictions on Training Set
        """        
        self.v_hat_saved = []
        self.delta_iters = None
        self.subgroup_updated_iters = None
        self.v_updated_iters = None
        self.verbose = verbose

    def fit(self, confs, labels, subgroups, params):
        """
        confs: initial confs on positive class
        labels: labels for each data point
        subgroups: (ordered) list of lists where each entry is a list of all indices of data belonging to 
                    a certain subgroup
        max_iter: max # iterations before terminating
        params: dictionary of hyperparameters
        """
        try:
            self.lmbda_type = params["lambda_type"]

            if self.lmbda_type == "uniform":
                self.lmbda = params["lambda"]
            elif self.lmbda_type == "range":
                self.lmbda_range = params["lambda_range"]
                assert len(self.lmbda_range) > 0
                assert self.lmbda_range[0] == 0 and self.lmbda_range[-1] == 1, "Lambda range must start at 0 and end at 1."
                assert all(self.lmbda_range[i] < self.lmbda_range[i+1] for i in range(len(self.lmbda_range)-1)), "Lambda range values must be strictly increasing."
            else:
                raise ValueError("Invalid lambda_type. Must be 'uniform' or 'range'.")

            self.alpha = params['alpha']
            self.max_iter = params['max_iter']
            self.randomized = params['randomized']
            self.use_oracle = params['use_oracle']
            self.tao_global = params["tao_global"] if "tao_global" in params else 1

        except KeyError as e:
            raise ValueError(f"Missing parameter: {e}. Please provide all required parameters (lambda, alpha, max_iter, randomized, use_oracle) as a dictionary.")

        # init predictions
        p = confs.copy()
        n = len(confs)
        alpha = self.alpha

        # count iterations
        iter = 0
        delta_iters = []
        subgroup_updated_iters = []
        v_updated_iters = []

        # get probability intervals and subgroups (including complements)
        if self.lmbda_type == "uniform":
            V_range = np.concatenate([np.arange(0, 1, self.lmbda), [1]])
        else:
            V_range = np.array(self.lmbda_range)

        C = [(i, sg) for i, sg in enumerate(subgroups)]

        # repeat until no updates made
        updated = True
        while updated and iter < self.max_iter:
            if self.verbose:
                print(f"Iteration {iter+1}...")
            updated = False
            iter += 1

            # track steps for test points
            delta = []
            subgroup_updated = []
            v_updated = []

            # shuffle subgroups and bins if randomized
            if self.randomized:
                np.random.shuffle(C)
                np.random.shuffle(V_range)

            # for each S in C, for each v in Lambda[0,1] (S_v := subgroup intersect v)
            for S_idx, S in C:
                # skip empty subgroups
                if (len(S) == 0): continue
                
                for v_i, v_j in zip(V_range, V_range[1:]):
                    S_v = [i for i in S if (v_i < p[i] <= v_j)]

                    # if subset size smaller than tao, throw out
                    tao = max(tao_global, alpha * len(S) / len(V_range))
                    if len(S_v) < tao:
                        continue

                    # retrieve offset from oracle
                    v_hat = np.mean(p[S_v]) # expected probability in S_v

                    if self.use_oracle:
                        r = self.oracle(subset=S_v, v_hat=v_hat, omega=(alpha/4), labels=labels)

                        # if no check, update predictions, projecting onto [0,1]
                        if r != 100:
                            p[S_v] = np.clip(p[S_v] + (r - v_hat), 0, 1)
                            updated = True

                            # update steps in procedure
                            delta.append(r - v_hat)
                            subgroup_updated.append(S_idx)
                            v_updated.append((v_i, v_j))
                    else:
                        dlta = np.mean(labels[S_v]) - v_hat
                        if (abs(dlta) < 1/(10 * len(V_range))):
                            continue
                        p[S_v] = np.clip(p[S_v] + dlta, 0, 1)
                        updated = True

                        # update steps in procedure
                        delta.append(dlta)
                        subgroup_updated.append(S_idx)
                        v_updated.append((v_i, v_j))

                    if self.verbose:
                        print(f"Updated uncertainty estimates for {len(S_v)} points in subgroup {S_idx} with v=({v_i}, {v_j})")

            delta_iters.append(delta)
            subgroup_updated_iters.append(subgroup_updated)
            v_updated_iters.append(v_updated)

            # save v_hats for current iteration
            self.v_hat_saved.append({})
            for v_i, v_j in zip(V_range, V_range[1:]):
                v_lmbda = [i for i in range(n) if v_i < p[i] <= v_j]

                # skip empty subgroups
                if (len(v_lmbda) == 0):
                    self.v_hat_saved[iter-1][(v_i, v_j)] = -1
                    continue

                v_hat = np.mean(p[v_lmbda])
                self.v_hat_saved[iter-1][(v_i, v_j)] = v_hat

        self.delta_iters = delta_iters
        self.subgroup_updated_iters = subgroup_updated_iters
        self.v_updated_iters = v_updated_iters

        return p

    # oracle: Guess and check SQ oracle to add noise
    def oracle(self, subset, v_hat, omega, labels):
        ps = np.mean(labels[subset])
        r=0
        
        # r == 100 indicates check
        if abs(ps-v_hat)<2*omega:
            r = 100
        if abs(ps-v_hat)>4*omega:
            r = np.random.uniform(0, 1)
        if r != 100:
            r = np.random.uniform(ps-omega, ps+omega)

        return r

    def _circuit_predict(self, f_x, subgroups_containing_x, early_stop=None):
        """
        Adjust Test-Set Predictions with Deltas from Multicalibration Procedure
            for $x \in X$:
            > for $lvl$ in circuit:
            >> if $x \in \lambda(v) \cap subgroup(lvl)$:
            >>> apply update (delta)
            >>
            >>> project to $[0,1]$ if needed
            >
            return predictions

        :param f_x: initial prediction (float)
        """
        # name vars
        early_stop = early_stop if early_stop else len(self.subgroup_updated_iters)
        mcb_pred = f_x.copy()
        subgroup_updated_iters = self.subgroup_updated_iters
        v_updated_iters = self.v_updated_iters
        delta_iters = self.delta_iters

        if self.lmbda_type == "uniform":
            V_range = np.concatenate([np.arange(0, 1, self.lmbda), [1]])
        else:
            V_range = np.array(self.lmbda_range)

        for subgroup_updated, v_updated, delta in zip(subgroup_updated_iters[:early_stop], v_updated_iters[:early_stop], delta_iters[:early_stop]):
            # for each lvl in circuit
            for lvl in range(len(subgroup_updated)):
                # check if datapoint belongs to $subgroup \cap lambda(v)$
                if subgroup_updated[lvl] in subgroups_containing_x:
                    (v_i, v_j) = v_updated[lvl]
                    if v_i < mcb_pred <= v_j:
                        # apply update, project onto [0, 1]
                        mcb_pred = np.clip(mcb_pred + delta[lvl], 0, 1)

        # get final prediciton from calib set v_hats
        for v_i, v_j in zip(V_range, V_range[1:]):
            if v_i < mcb_pred <= v_j:
                # if empty interval, return same prediction
                if self.v_hat_saved[-1][(v_i, v_j)] != -1:
                    mcb_pred = self.v_hat_saved[-1][(v_i, v_j)]
                break

        return mcb_pred

    def predict(self, f_xs, groups, early_stop=None):
        """
        :param f_x: initial prediction (float)
        """
        # name vars
        early_stop = early_stop if early_stop else len(self.subgroup_updated_iters)
        mcb_preds = f_xs.copy()

        if self.verbose:
            print(f"Predicting {len(f_xs)} data points...")
            range_func = trange
        else:
            range_func = range
        
        for i in range_func(len(f_xs)):
            mcb_preds[i] = self._circuit_predict(f_xs[i], [j for j in range(len(groups)) if i in groups[j]], early_stop=early_stop)

        return mcb_preds