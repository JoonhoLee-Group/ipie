#include "custom_types.h"

#include "gtest/gtest.h"

TEST(custom_types, operations) {
    ipie::new_energy_t x{1, 3, 2};
    ASSERT_EQ(x.e1b, ipie::complex_t{3});
    ASSERT_EQ(x.etot, ipie::complex_t{1});
    ASSERT_EQ(x.e2b, ipie::complex_t{2});
}

TEST(custom_types, add_assignment) {
    ipie::new_energy_t x{1, 3, -2};
    ipie::new_energy_t y{1, 3.3, -2.3};
    x += y;
    ASSERT_EQ(x.etot, ipie::complex_t{2});
    ASSERT_EQ(x.e1b, ipie::complex_t{6.3});
    ASSERT_EQ(x.e2b, ipie::complex_t{-4.3});
}

TEST(custom_types, scale) {
    ipie::new_energy_t x{1, 3, -2};
    double y = 4.0;
    x *= y;
    ASSERT_EQ(x.etot, ipie::complex_t{4});
    ASSERT_EQ(x.e1b, ipie::complex_t{4 * 3});
    ASSERT_EQ(x.e2b, ipie::complex_t{-2 * 4});
    ipie::new_energy_t xc{{1, 2}, {3, 4.7}, {-2, 3.3}};
    ipie::complex_t z{4.0, -2.0};
    xc *= z;
    EXPECT_NEAR(abs(xc.etot), abs((ipie::complex_t{8, 6})), 1e-12);
    EXPECT_NEAR(abs(xc.e1b), abs((ipie::complex_t{21.4, 12.8})), 1e-12);
    EXPECT_NEAR(abs(xc.e2b), abs((ipie::complex_t{-1.4, 17.2})), 1e-12);
}