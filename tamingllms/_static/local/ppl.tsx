import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ErrorBar } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const ModelComparison = () => {
  // Perplexity data with error margins
  const pplData = [
    { 
      model: 'Q2', 
      pplRatioPercent: (1.103587 - 1) * 100, 
      pplRatioError: 0.007783 * 100,
      pplDiff: 1.751667,
      pplDiffError: 0.146474
    },
    { 
      model: 'Q4', 
      pplRatioPercent: (1.035039 - 1) * 100,
      pplRatioError: 0.003969 * 100,
      pplDiff: 0.592510,
      pplDiffError: 0.071893
    },
    { 
      model: 'Q6', 
      pplRatioPercent: (1.009254 - 1) * 100,
      pplRatioError: 0.001784 * 100,
      pplDiff: 0.156488,
      pplDiffError: 0.031618
    },
  ];

  // KL divergence data
  const klData = [
    { model: 'Q2', mean: 0.111707, median: 0.074315 },
    { model: 'Q4', mean: 0.029804, median: 0.019842 },
    { model: 'Q6', mean: 0.003549, median: 0.002481 },
  ];

  const boldAxisStyle = {
    fontSize: '14px',
    fontWeight: 'bold'
  };

  const axisLabelStyle = {
    fontSize: '16px',
    fontWeight: 'bold'
  };

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Perplexity Comparison vs Base Model</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={pplData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" tick={boldAxisStyle} label={{ value: "Model", position: "bottom", style: axisLabelStyle }} />
                  <YAxis tick={boldAxisStyle} label={{ value: "PPL Ratio - 1 (%)", angle: -90, position: "insideLeft", style: axisLabelStyle }} />
                  <Tooltip formatter={(value) => value.toFixed(2) + '%'} />
                  <Bar 
                    dataKey="pplRatioPercent" 
                    name="PPL Ratio - 1 (%)" 
                    fill="#3eaf7c"
                  >
                    <ErrorBar dataKey="pplRatioError" width={4} strokeWidth={2} stroke="#000" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={pplData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" tick={boldAxisStyle} label={{ value: "Model", position: "bottom", style: axisLabelStyle }} />
                  <YAxis tick={boldAxisStyle} label={{ value: "PPL Difference", angle: -90, position: "insideLeft", style: axisLabelStyle }} />
                  <Tooltip />
                  <Bar 
                    dataKey="pplDiff" 
                    name="PPL Difference" 
                    fill="#3eaf7c"
                  >
                    <ErrorBar dataKey="pplDiffError" width={4} strokeWidth={2} stroke="#000" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>KL Divergence Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={klData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" tick={boldAxisStyle} label={{ value: "Model", position: "bottom", style: axisLabelStyle }} />
                <YAxis tick={boldAxisStyle} label={{ value: "KL Divergence", angle: -90, position: "insideLeft", style: axisLabelStyle }} />
                <Tooltip />
                <Legend verticalAlign="top" height={36} />
                <Line type="monotone" dataKey="mean" name="Mean" stroke="#3eaf7c" strokeWidth={3} />
                <Line type="monotone" dataKey="median" name="Median" stroke="#c9b6e4" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelComparison;