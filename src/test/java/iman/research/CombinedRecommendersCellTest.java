package iman.research;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CombinedRecommendersCellTest {

    CombinedRecommendersCell cells1, cells2;

    @BeforeEach
    void setUp() {
        cells1 = new CombinedRecommendersCell(3,4);
        cells1.addRate(1d);
        cells1.addRate(1.2d);
        cells1.addRate(3.12d);
        cells1.addRate(4.92d);
        cells1.addRate(5d);

        cells2 = new CombinedRecommendersCell(3,4);
        cells2.addRate(1d);
        cells2.addRate(1.2d);
        cells2.addRate(3.12d);
        cells2.addRate(4.92d);
        cells2.addRate(5d);
    }

    @Test
    void getMean() {
        assertEquals(3.048d, cells1.getMean(), 0.00001);
    }

    @Test
    void getVariance() {
        assertEquals(2.985856d, cells1.getVariance(), 0.00001);
    }

    @Test
    void equals() {

        assertNotEquals(cells1, cells2);
        cells2 = cells1;
        assertEquals(cells1, cells2);
    }

    @Test
    void compareTo() {
        assertEquals(cells1.compareTo(cells2), 0);
        cells2.addRate(1d);
        assertTrue(cells1.compareTo(cells2) < 0);
        assertTrue(cells2.compareTo(cells1) > 0);
    }
}