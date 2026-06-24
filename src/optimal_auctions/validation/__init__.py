"""Independent validation oracles for the auction approximation.

These compute optimal-mechanism revenue by methods entirely separate from the
cutting-plane LP (e.g. numerical integration), so agreement with
`OptimalAuctionApproximation` is a non-circular check.
"""

from optimal_auctions.validation.ebm import ExclusiveBuyerMechanism

__all__ = ["ExclusiveBuyerMechanism"]
