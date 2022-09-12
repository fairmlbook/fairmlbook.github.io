"""Computing fairness criteria on data."""

import math
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def first_index_above(array, value):
    """Find the smallest index i for which array[i] > value.

    If no such value exists, return len(array).
    """
    array = np.array(array)
    v = np.concatenate([array > value, np.ones_like(array[-1:])])
    return np.argmax(v, axis=0)


def trisearch_int(f, lo, hi, tol=1):
    """Trinary search: minimize f(x) over [lo, hi], to within +/-tol in x.

    Works assuming f is quasiconvex.
    """
    while hi - lo > tol:
        m1 = (2*lo + hi)//3
        m2 = (lo + 2*hi)//3
        if f(m1) < f(m2):
            hi = m2-1
        else:
            lo = m1+1
    return (hi + lo)//2


def trisearch(f, lo, hi, tol):
    """Trinary search: minimize f(x) over [lo, hi], to within +/-tol in x.

    Works assuming f is quasiconvex.
    """
    while hi - lo > tol:
        m1 = (2.*lo + hi)/3
        m2 = (lo + 2.*hi)/3
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    return (hi + lo)/2


class CriteriaData(object):
    """Class for computing fairness criteria of data."""

    def __init__(self, cdfs, performance, totals):
        """Create a CriteriaData instance from marginals.

        cdfs and performance are both dataframes, with index = score and columns being the groups.

        cdfs[group][score] = fraction of people of that group with score below that score.
        performance[group][score] = fraction of people with that group & score that succeed.
        totals[group] = total number of people of that group.

        totals can be either a dictionary or an array (if an array, in the same order as cdfs.columns)
        """
        self.cdfs = cdfs
        self.performance = performance
        self.columns = self.cdfs.columns
        if isinstance(totals, dict):
            totals = [totals[r] for r in self.columns]
        self.totals = np.array(totals)
        self.flipped = self.cdfs.index[0] > self.cdfs.index[-1]
        self.pdfs = self.get_pdfs()

    def get_pdfs(self):
        cdf_vs = np.concatenate([[np.zeros_like(self.cdfs.values[0])],self.cdfs.values])
        pdf_vs = (cdf_vs[1:]-cdf_vs[:-1])
        pdf = pd.DataFrame(pdf_vs, columns=self.columns, index=self.cdfs.index)
        return pdf

    @property
    def trisearch(self):
        if self.cdfs.index.dtype == 'int64':
            return trisearch_int
        else:
            return trisearch

    @classmethod
    def from_individuals(self, data, groups = None, binsize=0.025):
        """Recover CriteriaData instance from individual performances.

        data should be a dataframe with three columns: the group key, the
        predictor score, and the result.  Scores will be binned into
        bins of size binsize to compute the performance per group/score.

        groups is the set of group keys to use; it may be specified to
        restrict the set, or to define the order.
        """
        groupkey, score, result = data.columns
        if groups is None:
            groups = list(data.groupby(groupkey).mean().sort(score).index)[::-1]

        data = data.copy()
        data = data.sort(score)
        data[score] = (data[score] / binsize).astype(int)
        datasets = {}
        cdfs = pd.DataFrame(index=np.arange(min(data[score]), max(data[score])+1),
                            columns=groups)
        performance = cdfs.copy()
        cdfs = cdfs.notnull()*0
        totals = []
        for group in groups:
            grouped = data[data[groupkey] == group].groupby(score)
            num_per_score = grouped.count()[result]
            totals.append(num_per_score.sum())
            cdfs[group][num_per_score.index] = num_per_score.values
            cdfs[group] = cdfs[group].cumsum() * 1. / cdfs[group].sum()
            performance[group] = grouped.mean()
        cdfs.index = cdfs.index * binsize
        performance.index = performance.index * binsize
        performance = performance.interpolate().fillna(0)  #Remove nans
        return self(cdfs, performance, np.array(totals))


    def profit_cutoffs(self, target_rate, _=None):
        losses = np.cumsum(self.compute_area_slices(self.performance-target_rate))
        ans = {}
        for group in self.columns:
            last_bad_row = np.argmin(losses[group].values)
            try:
                cutoff = losses.index[last_bad_row + 1]
            except IndexError:
                cutoff = losses.index[-1] + (1e-9 if not self.flipped else -1e-9)
            ans[group] = cutoff
        return ans

    def compute_area_slices(self, pdfs=None, performance=None):
        if performance is None:
            performance = self.performance
        if pdfs is None:
            pdfs = self.pdfs
        perf = performance.values
        return  pdfs * perf

    def compute_area(self, cutoffs = None, performance=None):
        if performance is None:
            performance = self.performance
        if cutoffs is None:
            return self.compute_area_slices(performance=performance).sum(axis=0)
        else:
            ans = []
            for group in self.columns:
                area = self.compute_area_slices(self.pdfs[group].loc[cutoffs[group]:],
                                                performance[group].loc[cutoffs[group]:]).sum(axis=0)
                ans.append(area)
            return np.array(ans)

    def compute_curves(self):
        dfs = []
        for value in [self.performance, 1-self.performance]:
            area_slices = self.compute_area_slices(performance=value).values
            #area_slices = np.vstack([np.zeros_like(area_slices[0]), area_slices])
            under_curve = area_slices.sum(axis=0)
            fraction_excluded = area_slices.cumsum(axis=0) / under_curve
            dfs.append(pd.DataFrame(1-fraction_excluded, index=self.cdfs.index, columns=self.columns))
        return dfs

    def score_two_sided_profit(self, point, target_rate):
        profit = 0
        fraction_nondefaulters = self.compute_area()
        losses = (1 - fraction_nondefaulters) * point[1] * target_rate
        gains = fraction_nondefaulters * point[0] * (1 - target_rate)
        profit = (self.totals * (gains - losses)).sum()
        return profit

    def two_sided_optimum(self, target_rate):
        good_frac, bad_frac = self.compute_curves()
        polygons = [Delaunay(list(zip(good_frac[group], bad_frac[group]))) for group in self.columns]
        valid_points = []
        for poly in polygons:
            for p in poly.points:
                if all(poly2.find_simplex(p) != -1 for poly2 in polygons):
                    valid_points.append(p)
        valid_points = np.array(valid_points)
        result = (-1, None)
        for p in valid_points:
            score = self.score_two_sided_profit(p, target_rate)
            if score > result[0]:
                result = (score, p)
        return result[1]

    def _find_other_endpoint(self, curvex, curvey, i, p):
        if curvex[0] > curvex[-1]:
            curvex = curvex[::-1]
            curvey = curvey[::-1]
            i = len(curvex) - 1 - i
            invert = lambda j: len(curvex) - 1 - j
        else:
            invert = lambda j: j

        me = (curvex[i], curvey[i])
        if p[0] < me[0]:
            lst = range(i-1, -1, -1)
        else:
            lst = range(i+1, len(curvex), 1)
        for j in lst:
            they = (curvex[j], curvey[j])
            ratio = (p[0] - me[0]) / (they[0] - me[0])
            if ratio > 1:
                continue
            assert ratio > 0
            midy = ratio * (they[1] - me[1]) + me[1]
            if midy > p[1]:
                return invert(j)

    def two_sided_ranges(self, p):
        ans = {}
        good_frac, bad_frac = self.compute_curves()
        for group in self.columns:
            polygon = Delaunay(list(zip(good_frac[group], bad_frac[group])))
            vertices = polygon.simplices[polygon.find_simplex(p)]
            vert_to_pair = lambda v: (good_frac[group].index[v], bad_frac[group].values[v])
            vals = list(map(vert_to_pair, vertices))
            answer = (min(vals), max(vals))
            for v in vertices:
                if np.abs(p - polygon.points[v]).max() < 1e-3:
                    val = vert_to_pair(v)
                    answer = (val, val)
                    break
                j = self._find_other_endpoint(good_frac[group].values, bad_frac[group].values, v, p)
                if j is None:
                    continue

                vals = list(map(vert_to_pair, [v, j]))
                answer2 = (min(vals), max(vals))
                if abs(answer2[1][1] - answer2[0][1]) < abs(answer[1][1] - answer[0][1]):
                    answer = answer2
            ans[group] = (answer[0][0], answer[1][0])
        return ans

    def evaluate_opportunity(self, cutoffs):
        if not isinstance(cutoffs, dict):
            return np.array([cutoffs[0] for r in self.columns])
        return self.compute_area(cutoffs) / self.compute_area()

    def opportunity_cutoffs(self, target_rate, target_is_profit=False):
        if target_is_profit:
            target_rate = self.get_best_opportunity(target_rate)
        ans = {}
        cumulative = np.cumsum(self.compute_area_slices(), axis=0).values
        goal_area = cumulative[-1] * (1-target_rate)
        indices = first_index_above(cumulative, goal_area)
        cutoffs = self.cdfs.index[indices]
        return dict(zip(self.columns, cutoffs))

    def demographic_cutoffs(self, target_rate, target_is_profit=False):
        if target_is_profit:
            target_rate = self.get_best_demographic(target_rate)
        ans = {}
        for group in self.columns:
            index = first_index_above(self.cdfs[group].values, (1-target_rate))-1
            cutoff = self.cdfs.index[max(index, 0)]
            ans[group] = cutoff
        return ans

    def fixed_cutoffs(self, target_rate, target_is_profit=False):
        if target_is_profit:
            target_rate = self.get_best_fixed(target_rate)
        return {r:target_rate for r in self.columns}

    def compute_profit(self, cutoffs, target_rate):
        if isinstance(cutoffs, np.ndarray):
            # Hack, where we get the right answer for two-sided equality
            return self.score_two_sided_profit(cutoffs, target_rate)
        profits = self.compute_area(cutoffs, performance=self.performance - target_rate)
        return (profits * self.totals).sum()

    def efficiency(self, cutoffs, target_rate):
        return self.compute_profit(cutoffs, target_rate) / self.compute_profit(self.profit_cutoffs(target_rate), target_rate)

    def get_best_opportunity(self, target_rate):
        return trisearch(lambda t: -self.compute_profit(self.opportunity_cutoffs(t), target_rate), 0, 1, 1e-3)

    def get_best_fixed(self, target_rate):
        return self.trisearch(lambda t: -self.compute_profit({r:t for r in self.columns}, target_rate), self.cdfs.index.min(), self.cdfs.index.max(), 1e-3)

    def get_best_demographic(self, target_rate):
        return trisearch(lambda t: -self.compute_profit(self.demographic_cutoffs(t), target_rate), 0, 1, 1e-3)

    def coverage(self, cutoffs):
        total = 0
        covered = 0
        for i, group in enumerate(self.columns):
            if group not in cutoffs:
                continue
            total += self.totals[i]
            covered += self.totals[i] * (1 - self.cdfs[group][cutoffs[group]])
        return covered / total
