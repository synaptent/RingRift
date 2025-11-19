import { RATING_CONSTANTS } from '../../shared/types/user';

export class RatingService {
  /**
   * Calculate new Elo ratings for two players
   * @param playerRating Current rating of the player
   * @param opponentRating Current rating of the opponent
   * @param actualScore 1 for win, 0.5 for draw, 0 for loss
   * @param kFactor K-factor for rating adjustment (default: 32)
   * @returns The new rating for the player
   */
  static calculateNewRating(
    playerRating: number,
    opponentRating: number,
    actualScore: number,
    kFactor: number = RATING_CONSTANTS.K_FACTOR
  ): number {
    const expectedScore = this.getExpectedScore(playerRating, opponentRating);
    const ratingChange = Math.round(kFactor * (actualScore - expectedScore));

    let newRating = playerRating + ratingChange;

    // Clamp rating within bounds
    newRating = Math.max(
      RATING_CONSTANTS.MIN_RATING,
      Math.min(RATING_CONSTANTS.MAX_RATING, newRating)
    );

    return newRating;
  }

  /**
   * Calculate expected score based on rating difference
   * Formula: 1 / (1 + 10^((Rb - Ra) / 400))
   */
  static getExpectedScore(ratingA: number, ratingB: number): number {
    return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
  }

  /**
   * Calculate rating changes for a multiplayer game (e.g. 3 or 4 players)
   * Treats the game as a set of pairwise matches against all opponents
   */
  static calculateMultiplayerRatings(
    players: { id: string; rating: number; rank: number }[]
  ): Map<string, number> {
    const newRatings = new Map<string, number>();

    for (const player of players) {
      let totalExpectedScore = 0;
      let totalActualScore = 0;

      for (const opponent of players) {
        if (player.id === opponent.id) continue;

        // Calculate expected score against this opponent
        totalExpectedScore += this.getExpectedScore(player.rating, opponent.rating);

        // Calculate actual score based on rank (lower rank is better)
        // If ranks are equal, it's a draw (0.5)
        // If player rank < opponent rank, player won (1.0)
        // If player rank > opponent rank, player lost (0.0)
        if (player.rank < opponent.rank) {
          totalActualScore += 1;
        } else if (player.rank === opponent.rank) {
          totalActualScore += 0.5;
        }
      }

      // Calculate average K-factor adjustment
      // We use a slightly lower K-factor for multiplayer to reduce volatility
      const kFactor = RATING_CONSTANTS.K_FACTOR / (players.length - 1);

      const ratingChange = Math.round(kFactor * (totalActualScore - totalExpectedScore));
      let newRating = player.rating + ratingChange;

      newRating = Math.max(
        RATING_CONSTANTS.MIN_RATING,
        Math.min(RATING_CONSTANTS.MAX_RATING, newRating)
      );

      newRatings.set(player.id, newRating);
    }

    return newRatings;
  }
}
