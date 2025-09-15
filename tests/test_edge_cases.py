"""
Edge case tests for tax law reasoning scenarios.

Tests complex tax situations, boundary conditions, and corner cases
to ensure robust handling of diverse real-world scenarios.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxfix_musr.schema_generator import manual_case
from taxfix_musr.logic_tree import LogicTree
from taxfix_musr.models import Case, Fact, Rule, Node, RuleKind, NodeType


class TestTaxLawEdgeCases:
    """Test edge cases in tax law calculations."""
    
    def test_donation_cap_boundary(self):
        """Test donation cap at exact 10% boundary."""
        case = manual_case(
            case_id="donation_cap_boundary",
            facts={
                "salary": 50000,
                "donation": 5000,  # Exactly 10%
                "retirement_contribution": 0
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should allow full donation deduction
        assert result["allowable_donation"] == 5000
        assert result["taxable_income"] == 40000  # 50000 - 10000 (standard)
    
    def test_donation_over_cap(self):
        """Test donation exceeding 10% cap."""
        case = manual_case(
            case_id="donation_over_cap",
            facts={
                "salary": 40000,
                "donation": 8000,  # 20% of income, should be capped at 4000
                "retirement_contribution": 0
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should cap donation at 10%
        assert result["allowable_donation"] == 4000  # 10% of 40000
        assert result["taxable_income"] == 30000  # 40000 - 10000 (standard)
    
    def test_retirement_cap_boundary(self):
        """Test retirement contribution at exact €6,000 cap."""
        case = manual_case(
            case_id="retirement_cap_boundary",
            facts={
                "salary": 80000,
                "donation": 0,
                "retirement_contribution": 6000  # At cap
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should allow full retirement deduction
        assert result["retirement_deduction"] == 6000
        assert result["taxable_income"] == 64000  # 80000 - 10000 (standard) - 6000
    
    def test_retirement_over_cap(self):
        """Test retirement contribution exceeding €6,000 cap."""
        case = manual_case(
            case_id="retirement_over_cap",
            facts={
                "salary": 100000,
                "donation": 0,
                "retirement_contribution": 8000  # Over cap
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should cap retirement at 6000
        assert result["retirement_deduction"] == 6000
        assert result["taxable_income"] == 84000  # 100000 - 10000 (standard) - 6000
    
    def test_child_credit_phase_out_start(self):
        """Test child credit at start of phase-out (€90,000 AGI)."""
        case = manual_case(
            case_id="child_credit_phase_start",
            facts={
                "salary": 100000,  # Will result in AGI of 90000 after standard deduction
                "donation": 0,
                "retirement_contribution": 0,
                "children": 1
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # AGI should be 90000, so no phase-out yet
        assert result["agi"] == 90000
        assert result["child_credit"] == 2000  # Full credit
    
    def test_child_credit_partial_phase_out(self):
        """Test child credit with partial phase-out."""
        case = manual_case(
            case_id="child_credit_partial_phase",
            facts={
                "salary": 110000,  # AGI will be 100000 after standard deduction
                "donation": 0,
                "retirement_contribution": 0,
                "children": 1
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # AGI = 100000, phase-out = 5% * (100000 - 90000) = 5% * 10000 = 500
        assert result["agi"] == 100000
        assert result["child_credit"] == 1500  # 2000 - 500
    
    def test_child_credit_complete_phase_out(self):
        """Test child credit completely phased out."""
        case = manual_case(
            case_id="child_credit_complete_phase",
            facts={
                "salary": 150000,  # AGI will be 140000 after standard deduction
                "donation": 0,
                "retirement_contribution": 0,
                "children": 1
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # AGI = 140000, phase-out = 5% * (140000 - 90000) = 5% * 50000 = 2500 (exceeds 2000)
        assert result["agi"] == 140000
        assert result["child_credit"] == 0  # Completely phased out
    
    def test_multiple_children_phase_out(self):
        """Test phase-out with multiple children."""
        case = manual_case(
            case_id="multiple_children_phase",
            facts={
                "salary": 110000,  # AGI = 100000
                "donation": 0,
                "retirement_contribution": 0,
                "children": 3
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Base credit = 3 * 2000 = 6000
        # Phase-out = 5% * 10000 = 500
        # Final credit = 6000 - 500 = 5500
        assert result["child_credit"] == 5500
    
    def test_standard_vs_itemized_boundary(self):
        """Test boundary case where itemized equals standard deduction."""
        case = manual_case(
            case_id="deduction_boundary",
            facts={
                "salary": 100000,
                "donation": 4000,
                "retirement_contribution": 6000,  # Total itemized = 10000
                "children": 0
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should choose standard deduction (tie goes to standard)
        assert result["deduction_choice"] == 10000
        assert result["deduction_path"] == "standard"
        assert result["taxable_income"] == 90000
    
    def test_itemized_exceeds_standard(self):
        """Test case where itemized clearly exceeds standard."""
        case = manual_case(
            case_id="itemized_wins",
            facts={
                "salary": 80000,
                "donation": 8000,  # Capped at 8000
                "retirement_contribution": 6000,  # Total = 14000 > 10000
                "children": 0
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should choose itemized deduction
        assert result["deduction_choice"] == 14000
        assert result["deduction_path"] == "itemized"
        assert result["taxable_income"] == 66000
    
    def test_zero_income_case(self):
        """Test edge case with zero income."""
        case = manual_case(
            case_id="zero_income",
            facts={
                "salary": 0,
                "donation": 1000,  # Can't deduct more than income
                "retirement_contribution": 2000,
                "children": 1
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # With zero income, no deductions should apply
        assert result["gross_income"] == 0
        assert result["donation_deduction"] == 0  # Can't deduct from zero income
        assert result["taxable_income"] == 0  # Can't go negative
    
    def test_very_high_income(self):
        """Test very high income case."""
        case = manual_case(
            case_id="very_high_income",
            facts={
                "salary": 1000000,
                "freelance": 200000,
                "donation": 150000,  # Should be capped at 120000 (10% of 1.2M)
                "retirement_contribution": 10000,  # Should be capped at 6000
                "children": 5
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Validate caps are applied
        assert result["gross_income"] == 1200000
        assert result["donation_deduction"] == 120000  # 10% cap
        assert result["retirement_deduction"] == 6000  # 6000 cap
        
        # Child credits should be completely phased out
        assert result["child_credit"] == 0  # Way above phase-out threshold
    
    def test_complex_multi_income_case(self):
        """Test complex case with multiple income sources."""
        case = manual_case(
            case_id="multi_income_complex",
            facts={
                "salary": 75000,
                "freelance": 25000,  # Total income = 100000
                "donation": 8000,
                "retirement_contribution": 5000,
                "children": 2
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Validate calculations
        assert result["gross_income"] == 100000
        assert result["donation_deduction"] == 8000  # Within 10% cap (10000)
        assert result["retirement_deduction"] == 5000  # Within 6000 cap
        
        # Should choose itemized (13000 > 10000)
        assert result["deduction_choice"] == 13000
        assert result["deduction_path"] == "itemized"
        assert result["taxable_income"] == 87000
        
        # AGI = 87000, no child credit phase-out
        assert result["child_credit"] == 4000  # 2 children * 2000


class TestBoundaryConditions:
    """Test numerical boundary conditions and floating point edge cases."""
    
    def test_floating_point_precision(self):
        """Test calculations with floating point numbers."""
        case = manual_case(
            case_id="floating_point",
            facts={
                "salary": 33333.33,
                "donation": 3333.33,  # Exactly 10%
                "retirement_contribution": 1666.67,
                "children": 1
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should handle floating point calculations correctly
        assert abs(result["donation_deduction"] - 3333.33) < 0.01
        assert result["gross_income"] == 33333.33
    
    def test_rounding_edge_cases(self):
        """Test rounding in phase-out calculations."""
        # Create case where phase-out calculation involves rounding
        case = manual_case(
            case_id="rounding_test",
            facts={
                "salary": 90001,  # AGI will be 80001, phase-out = 0.05 * 1 = 0.05
                "donation": 0,
                "retirement_contribution": 0,
                "children": 1
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Phase-out should be minimal but credit should be reduced
        assert result["agi"] == 80001
        # Credit = 2000 - (0.05 * 1) = 1999.95, but should handle rounding
        assert 1999.0 <= result["child_credit"] <= 2000.0


class TestErrorHandling:
    """Test error handling and invalid input scenarios."""
    
    def test_negative_values_handling(self):
        """Test handling of negative input values."""
        # Note: This tests the logic tree's robustness
        # In practice, schema validation should prevent negative values
        try:
            case = manual_case(
                case_id="negative_values",
                facts={
                    "salary": -1000,  # Negative salary
                    "donation": 500,
                    "retirement_contribution": 1000,
                    "children": 1
                }
            )
            
            logic_tree = LogicTree(case)
            result = logic_tree.compute()
            
            # System should handle gracefully (exact behavior depends on implementation)
            assert "gross_income" in result
            
        except ValueError:
            # It's also acceptable for the system to reject negative values
            pass
    
    def test_extremely_large_values(self):
        """Test handling of extremely large values."""
        case = manual_case(
            case_id="extreme_values",
            facts={
                "salary": 999999999,  # Very large number
                "donation": 100000000,
                "retirement_contribution": 50000,
                "children": 1
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should handle large numbers without overflow
        assert result["gross_income"] == 999999999
        assert result["donation_deduction"] == 99999999.9  # 10% cap
        assert result["retirement_deduction"] == 6000  # Capped
    
    def test_zero_children_handling(self):
        """Test handling of zero children (no child credit)."""
        case = manual_case(
            case_id="no_children",
            facts={
                "salary": 50000,
                "donation": 2000,
                "retirement_contribution": 3000,
                "children": 0
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should handle zero children correctly
        assert result["child_credit"] == 0
        assert "children" not in result or result["children"] == 0


class TestRealWorldScenarios:
    """Test realistic, complex scenarios that might occur in practice."""
    
    def test_self_employed_high_earner(self):
        """Test scenario of self-employed high earner with complex deductions."""
        case = manual_case(
            case_id="self_employed_high",
            facts={
                "salary": 0,
                "freelance": 180000,  # High freelance income
                "donation": 25000,  # Should be capped at 18000
                "retirement_contribution": 8000,  # Should be capped at 6000
                "children": 2
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        assert result["gross_income"] == 180000
        assert result["donation_deduction"] == 18000  # 10% cap
        assert result["retirement_deduction"] == 6000  # Cap
        
        # Should choose itemized (24000 > 10000)
        assert result["deduction_path"] == "itemized"
        assert result["taxable_income"] == 156000
        
        # Child credit should be completely phased out (AGI way above 130000)
        assert result["child_credit"] == 0
    
    def test_middle_class_family_optimization(self):
        """Test middle-class family at optimization boundary."""
        case = manual_case(
            case_id="middle_class_optimization",
            facts={
                "salary": 85000,
                "donation": 8500,  # Exactly at 10% cap
                "retirement_contribution": 6000,  # At cap
                "children": 2
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Should optimize deductions perfectly
        assert result["donation_deduction"] == 8500
        assert result["retirement_deduction"] == 6000
        assert result["deduction_choice"] == 14500  # Itemized wins
        assert result["taxable_income"] == 70500
        
        # AGI = 70500, no phase-out for child credit
        assert result["child_credit"] == 4000  # 2 children
    
    def test_retirement_age_scenario(self):
        """Test scenario typical of retirement-age taxpayer."""
        case = manual_case(
            case_id="retirement_age",
            facts={
                "salary": 30000,  # Part-time work
                "donation": 5000,  # Generous giving
                "retirement_contribution": 6000,  # Max contribution
                "children": 0  # Adult children
            }
        )
        
        logic_tree = LogicTree(case)
        result = logic_tree.compute()
        
        # Donation should be capped at 3000 (10% of 30000)
        assert result["donation_deduction"] == 3000
        assert result["retirement_deduction"] == 6000
        
        # Should choose standard deduction (10000 > 9000)
        assert result["deduction_path"] == "standard"
        assert result["taxable_income"] == 20000
        assert result["child_credit"] == 0
