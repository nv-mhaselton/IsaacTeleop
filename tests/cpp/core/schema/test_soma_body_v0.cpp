// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/soma_body_v0_generated.h>

#include <cstddef>
#include <memory>
#include <type_traits>

#define VT(field) (field + 2) * 2
static_assert(core::SomaBodyPoseV0::VT_JOINT_ROTATIONS == VT(0));
static_assert(core::SomaBodyPoseV0::VT_GLOBAL_TRANSLATION == VT(1));
static_assert(core::SomaBodyPoseV0::VT_GLOBAL_TRANSLATION_IS_VALID == VT(2));
static_assert(core::SomaBodyPoseV0Record::VT_DATA == VT(0));
static_assert(core::SomaBodyPoseV0Record::VT_TIMESTAMP == VT(1));

#define TYPE(field) decltype(std::declval<core::SomaBodyPoseV0>().field())
static_assert(std::is_same_v<TYPE(joint_rotations), const core::SomaBodyJointRotationsV0*>);
static_assert(std::is_same_v<TYPE(global_translation), const core::Point*>);
static_assert(std::is_same_v<TYPE(global_translation_is_valid), bool>);

static_assert(std::is_trivially_copyable_v<core::SomaJointRotationV0>);
static_assert(std::is_trivially_copyable_v<core::SomaBodyJointRotationsV0>);
static_assert(sizeof(core::SomaBodyJointRotationsV0) == 77 * sizeof(core::SomaJointRotationV0));

static_assert(core::SomaBodyJointV0_HIPS == 0);
static_assert(core::SomaBodyJointV0_LEFT_SHOULDER == 11);
static_assert(core::SomaBodyJointV0_RIGHT_SHOULDER == 39);
static_assert(core::SomaBodyJointV0_LEFT_LEG == 67);
static_assert(core::SomaBodyJointV0_RIGHT_TOE_END == 76);
static_assert(core::SomaBodyJointV0_NUM_JOINTS == 77);

TEST_CASE("SOMA body v0 rotation array has the public layout", "[soma_body_v0][struct]")
{
    core::SomaBodyJointRotationsV0 rotations;

    REQUIRE(rotations.values()->size() == static_cast<size_t>(core::SomaBodyJointV0_NUM_JOINTS));
    CHECK((*rotations.values())[core::SomaBodyJointV0_HIPS]->axis_angle().x() == 0.0f);
    CHECK_FALSE((*rotations.values())[core::SomaBodyJointV0_RIGHT_TOE_END]->is_valid());

    const core::SomaJointRotationV0 hips(core::Point(0.1f, 0.2f, 0.3f), true);
    rotations.mutable_values()->Mutate(core::SomaBodyJointV0_HIPS, hips);

    CHECK((*rotations.values())[core::SomaBodyJointV0_HIPS]->axis_angle().y() == Catch::Approx(0.2f));
    CHECK((*rotations.values())[core::SomaBodyJointV0_HIPS]->is_valid());
}

TEST_CASE("SOMA body pose v0 round trips through FlatBuffers", "[soma_body_v0][flatbuffers]")
{
    core::SomaBodyPoseV0T pose;
    pose.joint_rotations = std::make_shared<core::SomaBodyJointRotationsV0>();
    pose.joint_rotations->mutable_values()->Mutate(
        core::SomaBodyJointV0_HEAD, core::SomaJointRotationV0(core::Point(0.4f, 0.5f, 0.6f), true));
    pose.global_translation = std::make_shared<core::Point>(1.0f, 2.0f, 3.0f);
    pose.global_translation_is_valid = true;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaBodyPoseV0::Pack(builder, &pose));

    const auto* decoded = flatbuffers::GetRoot<core::SomaBodyPoseV0>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_rotations() != nullptr);
    REQUIRE(decoded->global_translation() != nullptr);
    const auto* head = (*decoded->joint_rotations()->values())[core::SomaBodyJointV0_HEAD];
    CHECK(head->axis_angle().z() == Catch::Approx(0.6f));
    CHECK(head->is_valid());
    CHECK(decoded->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->global_translation_is_valid());
}
