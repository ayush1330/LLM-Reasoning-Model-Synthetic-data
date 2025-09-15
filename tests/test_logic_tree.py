"""
Tests for logic tree computation engine.

Tests topological ordering, safe expression evaluation, and rule application
to ensure ground truth computation works correctly.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxfix_musr.logic_tree import LogicTree
from taxfix_musr.schema_generator import manual_case


class TestLogicTree:
    """Test cases for LogicTree computation."""

    def test_known_case_computation(self):
        """Test computation of known case with expected result."""
        # Create case with known facts:
        #   - Salary: €80,000
        #   - Freelance: €15,000
        #   - Donations: €5,000 (under 10% cap of €95,000 = €9,500)
        #   - Retirement: €7,000 (exceeds €6,000 cap)
        # Expected: taxable_income = 95000 - 5000 - 6000 = 84000

        case = manual_case(
            case_id="test_case_1",
            facts={
                "salary": 80000,
                "freelance": 15000,
                "donation": 5000,
                "retirement_contribution": 7000,
            }
        )

        # Compute the case
        logic_tree = LogicTree(case)
        results = logic_tree.compute()

        # Verify individual computations
        assert results["salary"] == 80000
        assert results["freelance"] == 15000
        assert results["donation"] == 5000
        assert results["retirement_contribution"] == 7000

        # Verify gross income
        assert results["gross_income"] == 95000  # 80000 + 15000

        # Verify allowable donation (under 10% cap)
        assert results["allowable_donation"] == 5000  # min(5000, 95000 * 0.10)

        # Verify allowable retirement (capped at 6000)
        assert results["allowable_retirement"] == 6000  # min(7000, 6000)

        # Verify final taxable income
        expected_taxable = 95000 - 5000 - 6000  # 84000
        assert abs(results["taxable_income"] - expected_taxable) < 0.01

    def test_topological_sorting(self):
        """Test that nodes are sorted in correct dependency order."""
        case = manual_case(
            case_id="test_case_2",
            facts={
                "salary": 50000,
                "freelance": 10000,
                "donation": 3000,
                "retirement_contribution": 4000,
            }
        )

        logic_tree = LogicTree(case)
        ordered_nodes = logic_tree._topological_sort()

        # Facts should come first (no dependencies)
        fact_nodes = ["salary", "freelance", "donation", "retirement_contribution"]
        for fact in fact_nodes:
            assert fact in ordered_nodes

        # gross_income should come after salary and freelance
        gross_income_idx = ordered_nodes.index("gross_income")
        salary_idx = ordered_nodes.index("salary")
        freelance_idx = ordered_nodes.index("freelance")
        assert gross_income_idx > salary_idx
        assert gross_income_idx > freelance_idx

        # allowable_donation should come after donation and gross_income
        allowable_donation_idx = ordered_nodes.index("allowable_donation")
        donation_idx = ordered_nodes.index("donation")
        assert allowable_donation_idx > donation_idx
        assert allowable_donation_idx > gross_income_idx

        # taxable_income should come last
        assert ordered_nodes[-1] == "taxable_income"

    def test_safe_expression_eval(self):
        """Test safe mathematical expression evaluation."""
        case = manual_case(
            case_id="test_case_3",
            facts={
                "salary": 1000,
                "freelance": 2000,
                "donation": 100,
                "retirement_contribution": 200,
            }
        )

        logic_tree = LogicTree(case)

        # Test basic arithmetic
        assert logic_tree._safe_eval("1000 + 2000", {}) == 3000
        assert logic_tree._safe_eval("1000 - 200", {}) == 800
        assert logic_tree._safe_eval("1000 * 2", {}) == 2000
        assert logic_tree._safe_eval("1000 / 2", {}) == 500

        # Test with variables
        context = {"a": 1000, "b": 2000}
        assert logic_tree._safe_eval("a + b", context) == 3000

        # Test min/max functions
        assert logic_tree._safe_eval("min(1000, 2000)", {}) == 1000
        assert logic_tree._safe_eval("max(1000, 2000)", {}) == 2000

        # Test error handling
        with pytest.raises(ValueError, match="Undefined variable"):
            logic_tree._safe_eval("undefined_var", {})

        with pytest.raises(ValueError, match="Division by zero"):
            logic_tree._safe_eval("1000 / 0", {})

    def test_rule_application(self):
        """Test application of cap rules."""
        case = manual_case(
            case_id="test_case_4",
            facts={
                "salary": 50000,
                "freelance": 0,
                "donation": 8000,  # Exceeds 10% cap (5000)
                "retirement_contribution": 8000,  # Exceeds 6000 cap
            }
        )

        logic_tree = LogicTree(case)
        results = logic_tree.compute()

        # Test donation cap rule
        assert results["allowable_donation"] == 5000  # min(8000, 50000 * 0.10)

        # Test retirement cap rule
        assert results["allowable_retirement"] == 6000  # min(8000, 6000)

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        # Create a case with circular dependency
        facts = {
            "salary": 50000,
            "freelance": 10000,
            "donation": 3000,
            "retirement_contribution": 4000,
        }

        case = manual_case(
            case_id="test_case_5",
            facts=facts
        )

        # Manually create circular dependency
        case.nodes["gross_income"].depends_on = ["taxable_income"]  # Circular!

        logic_tree = LogicTree(case)

        with pytest.raises(ValueError, match="Circular dependency detected"):
            logic_tree._topological_sort()


@pytest.fixture
def sample_case():
    """Create a sample tax case for testing."""
    return manual_case(
        case_id="sample_case",
        facts={
            "salary": 60000,
            "freelance": 5000,
            "donation": 2000,
            "retirement_contribution": 3000,
        }
    )


@pytest.fixture
def known_result_case():
    """Create a case with known expected result."""
    return manual_case(
        case_id="known_result_case",
        facts={
            "salary": 80000,
            "freelance": 15000,
            "donation": 5000,
            "retirement_contribution": 7000,
        }
    )


def test_child_credit_phaseout_above_threshold():
    """Test child credit phase-out when AGI is slightly above €90,000."""
    case = manual_case(
        case_id="child_credit_test_1",
        facts={
            "salary": 85000,
            "freelance": 10000,
            "donation": 2000,
            "retirement_contribution": 3000,
        }
    )
    
    logic_tree = LogicTree(case)
    result = logic_tree.compute()
    
    # AGI = 85000 + 10000 - 2000 - 3000 = 90000
    # Child credit = 2000 (no phase-out since AGI = 90000)
    expected_agi = 90000
    expected_child_credit = 2000
    
    assert abs(result["AGI"] - expected_agi) < 1e-2
    assert abs(result["child_credit"] - expected_child_credit) < 1e-2


def test_child_credit_phaseout_well_above_threshold():
    """Test child credit phase-out when AGI is well above €90,000."""
    case = manual_case(
        case_id="child_credit_test_2",
        facts={
            "salary": 100000,
            "freelance": 20000,
            "donation": 5000,
            "retirement_contribution": 5000,
        }
    )
    
    logic_tree = LogicTree(case)
    result = logic_tree.compute()
    
    # AGI = 100000 + 20000 - 5000 - 5000 = 110000
    # Child credit = max(0, 2000 - 0.05 * (110000 - 90000)) = max(0, 2000 - 1000) = 1000
    expected_agi = 110000
    expected_child_credit = 1000
    
    assert abs(result["AGI"] - expected_agi) < 1e-2
    assert abs(result["child_credit"] - expected_child_credit) < 1e-2


def test_child_credit_phaseout_to_zero():
    """Test child credit phase-out when AGI is high enough to reduce credit to zero."""
    case = manual_case(
        case_id="child_credit_test_3",
        facts={
            "salary": 120000,
            "freelance": 25000,
            "donation": 10000,
            "retirement_contribution": 5000,
        }
    )
    
    logic_tree = LogicTree(case)
    result = logic_tree.compute()
    
    # AGI = 120000 + 25000 - 10000 - 5000 = 130000
    # Child credit = max(0, 2000 - 0.05 * (130000 - 90000)) = max(0, 2000 - 2000) = 0
    expected_agi = 130000
    expected_child_credit = 0
    
    assert abs(result["AGI"] - expected_agi) < 1e-2
    assert abs(result["child_credit"] - expected_child_credit) < 1e-2


def test_standard_deduction_chosen():
    """Test case where standard deduction is chosen over itemized."""
    case = manual_case(
        case_id="standard_deduction_test_1",
        facts={
            "salary": 50000,
            "freelance": 5000,
            "donation": 2000,  # Small donation
            "retirement_contribution": 3000,  # Small retirement contribution
        }
    )
    
    logic_tree = LogicTree(case)
    result = logic_tree.compute()
    
    # itemized_deduction = 2000 + 3000 = 5000
    # standard_deduction = 10000
    # Should choose standard (higher)
    expected_itemized = 5000
    expected_standard = 10000
    expected_deduction_choice = 10000
    expected_deduction_path = "standard"
    expected_taxable_income = 55000 - 10000  # gross_income - standard_deduction
    
    assert abs(result["itemized_deduction"] - expected_itemized) < 1e-2
    assert abs(result["standard_deduction_amount"] - expected_standard) < 1e-2
    assert abs(result["deduction_choice"] - expected_deduction_choice) < 1e-2
    assert result["deduction_path"] == expected_deduction_path
    assert abs(result["taxable_income"] - expected_taxable_income) < 1e-2


def test_itemized_deduction_chosen():
    """Test case where itemized deduction is chosen over standard."""
    case = manual_case(
        case_id="itemized_deduction_test_1",
        facts={
            "salary": 80000,
            "freelance": 20000,
            "donation": 8000,  # Large donation
            "retirement_contribution": 6000,  # Max retirement contribution
        }
    )
    
    logic_tree = LogicTree(case)
    result = logic_tree.compute()
    
    # itemized_deduction = 8000 + 6000 = 14000
    # standard_deduction = 10000
    # Should choose itemized (higher)
    expected_itemized = 14000
    expected_standard = 10000
    expected_deduction_choice = 14000
    expected_deduction_path = "itemized"
    expected_taxable_income = 100000 - 14000  # gross_income - itemized_deduction
    
    assert abs(result["itemized_deduction"] - expected_itemized) < 1e-2
    assert abs(result["standard_deduction_amount"] - expected_standard) < 1e-2
    assert abs(result["deduction_choice"] - expected_deduction_choice) < 1e-2
    assert result["deduction_path"] == expected_deduction_path
    assert abs(result["taxable_income"] - expected_taxable_income) < 1e-2
