import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const PerformanceComparison = () => {
  const performanceData = [
    { 
      model: 'Q2',
      tokens: 42.62,
      size: 390.28,
      logSize: Math.log10(390.28)
    },
    { 
      model: 'Q4',
      tokens: 38.38,
      size: 462.96,
      logSize: Math.log10(462.96)
    },
    { 
      model: 'Q6',
      tokens: 35.43,
      size: 614.58,
      logSize: Math.log10(614.58)
    },
    { 
      model: 'Base',
      tokens: 19.73,
      size: 1170,
      logSize: Math.log10(1170)
    }
  ];

  const boldAxisStyle = {
    fontSize: '14px',
    fontWeight: 'bold'
  };

  const axisLabelStyle = {
    fontSize: '16px',
    fontWeight: 'bold'
  };

  const formatLogAxis = (value) => {
    return Math.pow(10, value).toFixed(0);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Text Generation Performance vs Model Size</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="logSize" 
                tick={boldAxisStyle} 
                label={{ value: "Model Size (MiB)", position: "bottom", style: axisLabelStyle }}
                type="number"
                scale="linear"
                domain={['auto', 'auto']}
                tickFormatter={formatLogAxis}
              />
              <YAxis 
                tick={boldAxisStyle} 
                label={{ value: "Tokens/s", angle: -90, position: "insideLeft", style: axisLabelStyle }}
                domain={[15, 50]}
              />
              <Tooltip 
                formatter={(value, name, props) => {
                  if (name === "Tokens/s") {
                    return [`${value.toFixed(2)} ${name}`, props.payload.model];
                  }
                  return [value, name];
                }}
                labelFormatter={(value) => `Size: ${formatLogAxis(value)} MiB`}
              />
              <Line 
                type="monotone"
                dataKey="tokens" 
                name="Tokens/s" 
                stroke="#3eaf7c"
                strokeWidth={3}
                dot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default PerformanceComparison;