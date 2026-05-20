import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { runAdvancedAnalysis } from "@/services/pythonRunner";
import fs from "fs/promises";
import path from "path";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "JSON file is required" }, { status: 400 });
    }

    if (!file.name.toLowerCase().endsWith(".json")) {
      return NextResponse.json({ error: "Only .json files are supported" }, { status: 400 });
    }

    const rawText = await file.text();
    try {
      JSON.parse(rawText);
    } catch {
      return NextResponse.json({ error: "Invalid JSON file" }, { status: 400 });
    }

    const uploadDir = path.join(process.cwd(), "python", "uploads");
    await fs.mkdir(uploadDir, { recursive: true });

    const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
    const datasetPath = path.join(uploadDir, `${Date.now()}_${safeName}`);
    await fs.writeFile(datasetPath, rawText, "utf-8");

    const logs: string[] = [];
    const result = await runAdvancedAnalysis(datasetPath, (line) => {
      logs.push(line);
    });

    return NextResponse.json({ result, logs: logs.join("") });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
