export interface PolicyCheck {
  autoDecision: string | null;
  policyCheck: string;
  value: string;
  policySubparam: string | null;
}

export interface BureauDecisionResponse {
  status: number;
  data: {
    bureauDecision: string | null;
    loanAmount: number;
    decisionReason: string | null;
    decisionReasonList: string[] | null;
    bureauBasedMaxEligibleEmi: number | null;
    emiPlanList: any[] | null;
    mannualIntervention: string | null;
    policyChecks: PolicyCheck[];
    status: string | null;
  };
  attachment: any | null;
  message: string;
}

// Static bureau decision response
export const staticBureauResponse: BureauDecisionResponse = {
  status: 200,
  data: {
    bureauDecision: null,
    loanAmount: 12000,
    decisionReason: null,
    decisionReasonList: null,
    bureauBasedMaxEligibleEmi: null,
    emiPlanList: null,
    mannualIntervention: null,
    policyChecks: [
      {
        autoDecision: null,
        policyCheck: "Employment type",
        value: "SELF_EMPLOYED",
        policySubparam: null
      },
      {
        autoDecision: null,
        policyCheck: "Salary Verified",
        value: "true",
        policySubparam: null
      }
    ],
    status: null
  },
  attachment: null,
  message: "success"
}; 